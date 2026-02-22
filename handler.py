"""
RunPod serverless worker for Marker PDF conversion.

Input schema (job["input"]):
    pdf             - Required. Base64-encoded PDF/image bytes, or a URL to download the file from.
    filename        - Optional. Original filename (used for extension detection). Defaults to "document.pdf".
    page_range      - Optional. Page range string, e.g. "0-5". Defaults to all pages.
    force_ocr       - Optional. Force OCR even if text layer exists. Defaults to False.
    paginate_output - Optional. Add page delimiters to output. Defaults to False.
    output_format   - Optional. One of: "markdown", "json", "html", "chunks". Defaults to "markdown".
    use_llm         - Optional. Enable LLM-assisted conversion. Defaults to False.
    llm_service     - Optional. Fully-qualified LLM service class path.
                      Defaults to "marker.services.ollama.OllamaService".
                      Only used when use_llm=True.
    llm_config      - Optional. Dict of service-specific config passed directly to the service
                      constructor (e.g. {"ollama_model": "qwen3-vl:8b",
                      "ollama_base_url": "http://localhost:11434"}).
                      Only used when use_llm=True.

Output schema:
    success         - True on successful conversion.
    filename        - Original filename.
    output_format   - Format used for conversion.
    markdown        - Markdown text (when output_format="markdown").
    html            - HTML text (when output_format="html").
    json            - Structured JSON dict (when output_format="json").
    chunks          - Chunks text (when output_format="chunks").
    images          - Dict of image name -> base64-encoded PNG string
                      (populated for non-JSON output formats; empty for output_format="json").
    metadata        - Marker metadata dict.
    page_count      - Number of pages processed.

Environment variables:
    TORCH_DEVICE    - Device for inference ("cuda" or "cpu"). Defaults to "cuda".
    MODEL_CACHE_DIR - Directory where Marker/Surya models are downloaded and cached.
                      Set this to a persistent volume mount path (e.g. /runpod-volume/models)
                      so models survive container restarts. Defaults to /models (see Dockerfile).
"""

import base64
import gc
import io
import logging
import os
import tempfile
from pathlib import Path
from typing import Optional

import requests
import runpod

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

TORCH_DEVICE = os.environ.get("TORCH_DEVICE", "cuda")
os.environ.setdefault("TORCH_DEVICE", TORCH_DEVICE)

# MODEL_CACHE_DIR is read directly from the environment by the surya/marker
# pydantic-settings singleton at import time. Set it before importing marker.
# Defaults to /models (see Dockerfile); override to a persistent volume path
# (e.g. /runpod-volume/models) to avoid re-downloading models on cold starts.

ALLOWED_EXTENSIONS = {".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"}
VALID_OUTPUT_FORMATS = {"markdown", "json", "html", "chunks"}

# ---------------------------------------------------------------------------
# Model loading – executed once when the container starts.
# ---------------------------------------------------------------------------

logger.info("Loading Marker models...")
try:
    from marker.models import create_model_dict

    MODELS = create_model_dict()
    logger.info("Marker models loaded successfully (%d models).", len(MODELS))
except Exception:
    logger.exception("Failed to load Marker models.")
    MODELS = None


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _resolve_file(pdf_input: str, filename: str) -> bytes:
    """Return raw bytes from a base64 string or a URL."""
    if pdf_input.startswith(("http://", "https://")):
        logger.info("Downloading file from URL: %s", pdf_input)
        response = requests.get(pdf_input, timeout=120)
        response.raise_for_status()
        return response.content
    # Assume base64-encoded bytes
    return base64.b64decode(pdf_input)


def _build_llm_service(config_parser, llm_service: Optional[str], llm_config: Optional[dict]):
    """Return an LLM service instance, optionally overriding the class and config."""
    if llm_service:
        import importlib
        module_path, class_name = llm_service.rsplit(".", 1)
        module = importlib.import_module(module_path)
        service_cls = getattr(module, class_name)
        return service_cls(llm_config)
    # Fall back to whatever the config parser resolves.
    return config_parser.get_llm_service()


# ---------------------------------------------------------------------------
# RunPod handler
# ---------------------------------------------------------------------------

def handler(job: dict) -> dict:
    """Process a single PDF conversion job."""
    job_input: dict = job.get("input", {})

    # --- validate models ---
    if MODELS is None:
        return {"success": False, "error": "Marker models failed to load. Check container logs."}

    # --- required field ---
    pdf_input: Optional[str] = job_input.get("pdf")
    if not pdf_input:
        return {"success": False, "error": "Missing required field: 'pdf' (base64 string or URL)."}

    # --- optional fields ---
    filename: str = job_input.get("filename", "document.pdf")
    page_range: Optional[str] = job_input.get("page_range")
    force_ocr: bool = bool(job_input.get("force_ocr", False))
    paginate_output: bool = bool(job_input.get("paginate_output", False))
    output_format: str = job_input.get("output_format", "markdown")
    use_llm: bool = bool(job_input.get("use_llm", False))
    llm_service_path: Optional[str] = job_input.get("llm_service")
    llm_config: Optional[dict] = job_input.get("llm_config")

    if use_llm and not llm_service_path:
        llm_service_path = "marker.services.ollama.OllamaService"

    if llm_config is not None and not isinstance(llm_config, dict):
        return {"success": False, "error": "'llm_config' must be a JSON object (dict)."}

    # --- validate extension ---
    file_ext = Path(filename).suffix.lower() or ".pdf"
    if file_ext not in ALLOWED_EXTENSIONS:
        return {
            "success": False,
            "error": f"Unsupported file type '{file_ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        }

    # --- validate output format ---
    if output_format not in VALID_OUTPUT_FORMATS:
        return {
            "success": False,
            "error": f"Invalid output_format '{output_format}'. Must be one of: {sorted(VALID_OUTPUT_FORMATS)}",
        }

    # --- resolve file bytes ---
    try:
        file_bytes = _resolve_file(pdf_input, filename)
    except Exception as exc:
        logger.exception("Failed to retrieve file.")
        return {"success": False, "error": f"Failed to retrieve file: {exc}"}

    # --- write to a temp file and convert ---
    # temp_path starts as None so the finally clause is safe even if the
    # NamedTemporaryFile context manager raises before the assignment.
    temp_path = None
    try:
        with tempfile.NamedTemporaryFile(suffix=file_ext, delete=False) as tmp:
            tmp.write(file_bytes)
            temp_path = tmp.name

        from marker.config.parser import ConfigParser
        from marker.converters.pdf import PdfConverter
        from marker.settings import settings

        config = {
            "filepath": temp_path,
            "page_range": page_range,
            "force_ocr": force_ocr,
            "paginate_output": paginate_output,
            "output_format": output_format,
            "use_llm": use_llm,
        }

        config_parser = ConfigParser(config)
        config_dict = config_parser.generate_config_dict()
        config_dict["pdftext_workers"] = 1

        llm_service_instance = (
            _build_llm_service(config_parser, llm_service_path, llm_config)
            if use_llm
            else None
        )

        converter = PdfConverter(
            config=config_dict,
            artifact_dict=MODELS,
            processor_list=config_parser.get_processors(),
            renderer=config_parser.get_renderer(),
            llm_service=llm_service_instance,
        )

        logger.info("Converting '%s' to %s …", filename, output_format)
        rendered_output = converter(temp_path)

        # --- extract content ---
        json_content = None
        html_content = None
        markdown_content = None
        chunks_content = None
        encoded_images: dict = {}

        if output_format == "json":
            json_content = rendered_output.model_dump()
        else:
            from marker.output import text_from_rendered

            text, _, images = text_from_rendered(rendered_output)

            if output_format == "html":
                html_content = text
            elif output_format == "chunks":
                chunks_content = text
            else:
                markdown_content = text

            for img_name, img_obj in images.items():
                buf = io.BytesIO()
                img_obj.save(buf, format=settings.OUTPUT_IMAGE_FORMAT)
                encoded_images[img_name] = base64.b64encode(buf.getvalue()).decode("utf-8")

        metadata = rendered_output.metadata
        logger.info("Conversion of '%s' completed successfully.", filename)

        return {
            "success": True,
            "filename": filename,
            "output_format": output_format,
            "markdown": markdown_content,
            "html": html_content,
            "json": json_content,
            "chunks": chunks_content,
            "images": encoded_images,
            "metadata": metadata,
            "page_count": len(metadata.get("page_stats", [])),
        }

    except Exception as exc:
        logger.exception("Conversion failed for '%s'.", filename)
        return {"success": False, "error": f"Conversion failed: {exc}"}

    finally:
        if temp_path and os.path.exists(temp_path):
            os.unlink(temp_path)
        gc.collect()


if __name__ == "__main__":
    runpod.serverless.start({"handler": handler})
