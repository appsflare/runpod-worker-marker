"""
Manages the lifecycle of a local Ollama server process for use within the RunPod handler.

A single ``OllamaRunner`` instance should be held at module level in the handler so the
Ollama server persists across multiple RunPod jobs (warm-start reuse).

Environment variables:
    OLLAMA_READY_TIMEOUT - Seconds to wait for Ollama to become ready after start.
                           Defaults to 60.
"""

import logging
import os
import signal
import subprocess
import time
from typing import Optional

import requests

logger = logging.getLogger(__name__)

OLLAMA_READY_TIMEOUT = int(os.environ.get("OLLAMA_READY_TIMEOUT", "60"))


class OllamaRunner:
    """Manages a local ``ollama serve`` background process.

    Typical usage::

        # module level – shared across all jobs
        ollama_runner = OllamaRunner()

        # inside a job handler
        ollama_runner.ensure_ready(base_url="http://localhost:11434", model="qwen2.5vl:7b")

        # when a stop_ollama action is received
        ollama_runner.stop()
    """

    def __init__(self) -> None:
        self._process: Optional[subprocess.Popen] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def is_ollama_service(llm_service: str) -> bool:
        """Return ``True`` if *llm_service* refers to an OllamaService class."""
        return "ollamaservice" in llm_service.lower()

    def start(self, base_url: str = "http://localhost:11434") -> None:
        """Start ``ollama serve`` in the background if not already running, then wait until ready."""
        if self._process is not None and self._process.poll() is None:
            logger.info("Ollama already running (pid %d), skipping start.", self._process.pid)
            return

        logger.info("Starting ollama serve in background…")
        self._process = subprocess.Popen(
            ["ollama", "serve"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
        logger.info("Ollama process started (pid %d).", self._process.pid)
        self._wait_until_ready(base_url)

    def pull_model(self, model: str, base_url: str = "http://localhost:11434") -> None:
        """Pull *model* if it is not already present in the local Ollama registry.

        The pull is blocking so the model is fully available before the job proceeds.
        """
        if not model:
            logger.warning("No ollama_model specified in llm_config; skipping pull.")
            return

        if self._model_present(model, base_url):
            logger.info("Model '%s' already present, skipping pull.", model)
            return

        logger.info("Pulling model '%s' …", model)
        subprocess.run(["ollama", "pull", model], check=True)
        logger.info("Model '%s' ready.", model)

    def ensure_ready(
        self,
        base_url: str = "http://localhost:11434",
        model: Optional[str] = None,
    ) -> None:
        """Convenience method: start Ollama, then pull the requested model if given."""
        self.start(base_url)
        if model:
            self.pull_model(model, base_url)

    def stop(self) -> None:
        """Terminate the Ollama background process gracefully (SIGTERM → SIGKILL)."""
        if self._process is None or self._process.poll() is not None:
            logger.info("Ollama is not running; nothing to stop.")
            self._process = None
            return

        logger.info("Stopping Ollama (pid %d) …", self._process.pid)
        self._process.send_signal(signal.SIGTERM)
        try:
            self._process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("Ollama did not exit after SIGTERM; sending SIGKILL.")
            self._process.kill()
            self._process.wait()

        self._process = None
        logger.info("Ollama process stopped.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _wait_until_ready(self, base_url: str) -> None:
        """Poll ``GET /api/tags`` until Ollama responds or *OLLAMA_READY_TIMEOUT* is exceeded."""
        deadline = time.monotonic() + OLLAMA_READY_TIMEOUT
        attempt = 0
        while time.monotonic() < deadline:
            attempt += 1
            try:
                resp = requests.get(f"{base_url}/api/tags", timeout=2)
                if resp.status_code == 200:
                    logger.info("Ollama is ready (attempt %d).", attempt)
                    return
            except requests.exceptions.RequestException:
                pass
            logger.info("Waiting for Ollama to be ready… attempt %d", attempt)
            time.sleep(2)

        raise RuntimeError(
            f"Ollama did not become ready within {OLLAMA_READY_TIMEOUT} seconds."
        )

    @staticmethod
    def _model_present(model: str, base_url: str) -> bool:
        """Return ``True`` if *model* is already in the local Ollama registry."""
        try:
            resp = requests.get(f"{base_url}/api/tags", timeout=10)
            resp.raise_for_status()
            models = resp.json().get("models", [])
            return any(m.get("name") == model for m in models)
        except Exception:
            return False
