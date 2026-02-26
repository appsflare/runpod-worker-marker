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
import threading
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
        self._lock = threading.RLock()
        self._state_cv = threading.Condition(self._lock)
        self._start_in_progress = False
        self._stop_in_progress = False
        self._stop_requested = False
        self._last_start_error: Optional[Exception] = None

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    @staticmethod
    def is_ollama_service(llm_service: str) -> bool:
        """Return ``True`` if *llm_service* refers to an OllamaService class."""
        return "ollamaservice" in llm_service.lower()

    def start(self, base_url: str = "http://localhost:11434") -> None:
        """Start ``ollama serve`` in the background if not already running, then wait until ready."""
        with self._state_cv:
            while self._stop_in_progress:
                logger.info("Stop in progress; waiting before start.")
                self._state_cv.wait()

            if self._process is not None and self._process.poll() is None:
                logger.info("Ollama already running (pid %d), skipping start.", self._process.pid)
                return

            if self._start_in_progress:
                logger.info("Ollama start is pending; waiting for in-flight startup to finish.")
                while self._start_in_progress:
                    self._state_cv.wait()

                if self._process is not None and self._process.poll() is None:
                    logger.info("In-flight startup completed; Ollama is running.")
                    return

                if self._last_start_error is not None:
                    raise RuntimeError("Ollama startup failed in another request.") from self._last_start_error

                raise RuntimeError("Ollama startup did not complete successfully.")

            if self._stop_requested:
                raise RuntimeError("Ollama startup canceled because stop was requested.")

            self._start_in_progress = True
            self._last_start_error = None

        start_error: Optional[Exception] = None
        process: Optional[subprocess.Popen] = None
        try:
            logger.info("Starting ollama serve in background…")
            process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
            logger.info("Ollama process started (pid %d).", process.pid)

            with self._state_cv:
                self._process = process
                self._state_cv.notify_all()

            self._wait_until_ready(base_url, process)
        except Exception as exc:
            start_error = exc
            raise
        finally:
            with self._state_cv:
                self._start_in_progress = False
                self._last_start_error = start_error

                if self._process is not None and self._process.poll() is not None:
                    self._process = None

                self._state_cv.notify_all()

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
        with self._state_cv:
            self._stop_requested = True

            while self._stop_in_progress:
                self._state_cv.wait()

            self._stop_in_progress = True
            process = self._process

            while self._start_in_progress and (process is None or process.poll() is not None):
                self._state_cv.wait(timeout=0.1)
                process = self._process

        if process is None or process.poll() is not None:
            logger.info("Ollama is not running; nothing to stop.")
            with self._state_cv:
                self._process = None
                self._stop_in_progress = False
                self._stop_requested = False
                self._state_cv.notify_all()
            return

        logger.info("Stopping Ollama (pid %d) …", process.pid)
        process.send_signal(signal.SIGTERM)
        try:
            process.wait(timeout=10)
        except subprocess.TimeoutExpired:
            logger.warning("Ollama did not exit after SIGTERM; sending SIGKILL.")
            process.kill()
            process.wait()

        with self._state_cv:
            self._process = None
            self._stop_in_progress = False
            self._stop_requested = False
            self._state_cv.notify_all()

        logger.info("Ollama process stopped.")

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _wait_until_ready(self, base_url: str, process: subprocess.Popen) -> None:
        """Poll ``GET /api/tags`` until Ollama responds or *OLLAMA_READY_TIMEOUT* is exceeded."""
        deadline = time.monotonic() + OLLAMA_READY_TIMEOUT
        attempt = 0
        while time.monotonic() < deadline:
            attempt += 1

            with self._state_cv:
                if self._stop_requested:
                    raise RuntimeError("Ollama startup canceled due to stop request.")

            if process.poll() is not None:
                raise RuntimeError("Ollama process exited before becoming ready.")

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
