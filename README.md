# Runpod-Worker-Marker

A [RunPod](https://www.runpod.io/) serverless worker that converts PDFs and images to Markdown, HTML, JSON, or chunks using [Marker](https://github.com/datalab-to/marker).

---

## Features

- Converts PDF, PNG, JPEG, TIFF, and BMP files.
- Output formats: `markdown`, `html`, `json`, `chunks`.
- Optional LLM-assisted conversion with **your choice of LLM service and model**.
- GPU-accelerated via CUDA.
- Dependency management via **[UV](https://github.com/astral-sh/uv)**.

---

## Local Development

### Prerequisites

- [UV](https://docs.astral.sh/uv/getting-started/installation/) installed.
- Python 3.10+.
- (Optional) CUDA-capable GPU.

### Setup

```bash
uv sync
```

This reads `pyproject.toml` and installs all dependencies (including PyTorch with CUDA 12.1 wheels).

---

## Building the Docker Image

```bash
docker build -t runpod-worker-marker:latest .
```

---

## Deploying on RunPod

1. Push the image to a container registry (Docker Hub, GHCR, etc.).
2. Create a new **Serverless** endpoint in the RunPod console.
3. Set the container image to your pushed image.
4. (Optional) Set the `TORCH_DEVICE` environment variable to `cuda` (default) or `cpu`.

---

## Input Schema

Send a JSON payload to your endpoint:

| Field            | Type    | Required | Default      | Description                                                                                         |
|------------------|---------|----------|--------------|-----------------------------------------------------------------------------------------------------|
| `pdf`            | string  | ✅       | —            | Base64-encoded file bytes **or** a public URL to download the file from.                            |
| `filename`       | string  | ❌       | `document.pdf` | Original filename; used for file-type detection.                                                  |
| `page_range`     | string  | ❌       | all pages    | Page range, e.g. `"0-5"`.                                                                           |
| `force_ocr`      | boolean | ❌       | `false`      | Force OCR even when a text layer is present.                                                        |
| `paginate_output`| boolean | ❌       | `false`      | Insert page delimiters into the output.                                                             |
| `output_format`  | string  | ❌       | `"markdown"` | One of `"markdown"`, `"html"`, `"json"`, `"chunks"`.                                               |
| `use_llm`        | boolean | ❌       | `false`      | Enable LLM-assisted conversion.                                                                     |
| `llm_service`    | string  | ❌       | —            | Fully-qualified LLM service class, e.g. `"marker.services.claude.ClaudeService"`. Requires `use_llm=true`. |
| `llm_model`      | string  | ❌       | —            | Model identifier for the chosen service, e.g. `"claude-opus-4-5"`. Requires `use_llm=true`.       |

### Example — Markdown (base64 input)

```json
{
  "input": {
    "pdf": "<base64-encoded PDF bytes>",
    "filename": "report.pdf",
    "output_format": "markdown"
  }
}
```

### Example — HTML via URL

```json
{
  "input": {
    "pdf": "https://example.com/document.pdf",
    "filename": "document.pdf",
    "output_format": "html"
  }
}
```

### Example — LLM-assisted with a custom service and model

```json
{
  "input": {
    "pdf": "<base64-encoded PDF bytes>",
    "filename": "report.pdf",
    "output_format": "markdown",
    "use_llm": true,
    "llm_service": "marker.services.claude.ClaudeService",
    "llm_model": "claude-opus-4-5"
  }
}
```

---

## Output Schema

| Field          | Type            | Description                                                     |
|----------------|-----------------|-----------------------------------------------------------------|
| `success`      | boolean         | `true` on successful conversion.                                |
| `filename`     | string          | Original filename.                                              |
| `output_format`| string          | Format used.                                                    |
| `markdown`     | string \| null  | Markdown text (when `output_format="markdown"`).                |
| `html`         | string \| null  | HTML text (when `output_format="html"`).                        |
| `json`         | object \| null  | Structured data (when `output_format="json"`).                  |
| `chunks`       | string \| null  | Chunks text (when `output_format="chunks"`).                    |
| `images`       | object          | Map of image name → base64-encoded PNG string.                  |
| `metadata`     | object          | Marker metadata (language, page stats, etc.).                   |
| `page_count`   | integer         | Number of pages processed.                                      |
| `error`        | string          | Present only on failure; describes what went wrong.             |

