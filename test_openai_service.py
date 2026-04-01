"""
Unit tests for OpenAIServiceWithExtraBody and its integration with the handler.

Run with:
    python -m pytest test_openai_service.py -v

No GPU or running backend required — the OpenAI client and Marker model
loading are mocked.
"""

import json
import sys
import types
import unittest
from unittest.mock import MagicMock, patch


# ---------------------------------------------------------------------------
# Stub heavy modules that are imported at *module load time* in handler.py
# before we import the handler.  The real marker package is available in
# .venv and is left untouched so openai_service.py can use its real imports.
# ---------------------------------------------------------------------------

def _stub(name, **attrs):
    mod = types.ModuleType(name)
    for k, v in attrs.items():
        setattr(mod, k, v)
    sys.modules[name] = mod
    return mod


# handler.py does `from marker.models import create_model_dict` at the top
# level.  Stub only that symbol so the real marker package is otherwise intact.
_fake_models = {"layout": MagicMock(), "ocr": MagicMock()}
import marker.models as _real_marker_models  # noqa: E402  (real package from .venv)
_orig_create_model_dict = _real_marker_models.create_model_dict
_real_marker_models.create_model_dict = lambda: _fake_models

# ollama_runner lives in the project root and starts a subprocess; stub it.
_FakeOllamaRunner = MagicMock()
_FakeOllamaRunner.return_value.stop = MagicMock()
_FakeOllamaRunner.is_ollama_service = staticmethod(lambda path: "ollama" in path.lower())
_stub("ollama_runner", OllamaRunner=_FakeOllamaRunner)

# runpod is not installed locally; stub it.
_stub("runpod", serverless=MagicMock())

# ---------------------------------------------------------------------------
# Now import the real modules under test.
# ---------------------------------------------------------------------------

from openai_service import OpenAIServiceWithExtraBody  # noqa: E402
import handler as _handler_mod  # noqa: E402  (triggers module-level model load)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_openai_response(content: dict):
    resp = MagicMock()
    resp.choices[0].message.content = json.dumps(content)
    resp.usage.total_tokens = 42
    return resp


# ---------------------------------------------------------------------------
# Tests: OpenAIServiceWithExtraBody field assignment
# ---------------------------------------------------------------------------

class TestFieldAssignment(unittest.TestCase):

    def _make_service(self, extra_body=None):
        config = {
            "openai_base_url": "http://localhost:8000/v1",
            "openai_api_key": "EMPTY",
            "openai_model": "test-model",
        }
        if extra_body is not None:
            config["openai_extra_body"] = extra_body
        return OpenAIServiceWithExtraBody(config)

    def test_default_extra_body_is_empty_dict(self):
        svc = self._make_service()
        self.assertEqual(svc.openai_extra_body, {})

    def test_extra_body_assigned_from_config(self):
        body = {"top_k": 20, "min_p": 0.05}
        svc = self._make_service(extra_body=body)
        self.assertEqual(svc.openai_extra_body, body)

    def test_base_fields_still_assigned(self):
        svc = self._make_service()
        self.assertEqual(svc.openai_model, "test-model")
        self.assertEqual(svc.openai_base_url, "http://localhost:8000/v1")


# ---------------------------------------------------------------------------
# Tests: __call__ — extra_body forwarded to SDK
# ---------------------------------------------------------------------------

class TestCallExtraBody(unittest.TestCase):

    def _make_service(self, extra_body=None):
        config = {
            "openai_base_url": "http://localhost:8000/v1",
            "openai_api_key": "EMPTY",
            "openai_model": "test-model",
        }
        if extra_body is not None:
            config["openai_extra_body"] = extra_body
        return OpenAIServiceWithExtraBody(config)

    def test_extra_body_forwarded_to_sdk(self):
        body = {"chat_template_kwargs": {"enable_thinking": False}, "top_k": 50}
        svc = self._make_service(extra_body=body)

        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.return_value = _make_openai_response({"result": "ok"})

        with patch.object(svc, "get_client", return_value=mock_client):
            result = svc(
                prompt="Describe this.",
                image=None,
                block=None,
                response_schema=MagicMock(__name__="Schema"),
            )

        self.assertEqual(result, {"result": "ok"})
        _, kwargs = mock_client.beta.chat.completions.parse.call_args
        self.assertEqual(kwargs["extra_body"], body)

    def test_no_extra_body_sends_empty_dict(self):
        """When no extra_body is configured, an empty dict is still forwarded."""
        svc = self._make_service()
        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.return_value = _make_openai_response({})

        with patch.object(svc, "get_client", return_value=mock_client):
            svc(prompt="test", image=None, block=None, response_schema=MagicMock(__name__="S"))

        _, kwargs = mock_client.beta.chat.completions.parse.call_args
        self.assertEqual(kwargs["extra_body"], {})

    def test_model_and_timeout_passed(self):
        svc = self._make_service(extra_body={"top_k": 5})
        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.return_value = _make_openai_response({})

        with patch.object(svc, "get_client", return_value=mock_client):
            svc(prompt="test", image=None, block=None, response_schema=MagicMock(__name__="S"),
                timeout=10)

        _, kwargs = mock_client.beta.chat.completions.parse.call_args
        self.assertEqual(kwargs["model"], "test-model")
        self.assertEqual(kwargs["timeout"], 10)

    def test_retries_on_rate_limit(self):
        from openai import RateLimitError

        svc = self._make_service()
        svc.max_retries = 1
        svc.retry_wait_time = 0

        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.side_effect = [
            RateLimitError("rate limit", response=MagicMock(), body={}),
            _make_openai_response({"ok": True}),
        ]

        with patch.object(svc, "get_client", return_value=mock_client), patch("time.sleep"):
            result = svc(prompt="test", image=None, block=None,
                         response_schema=MagicMock(__name__="S"))

        self.assertEqual(result, {"ok": True})
        self.assertEqual(mock_client.beta.chat.completions.parse.call_count, 2)

    def test_returns_empty_dict_on_unhandled_exception(self):
        svc = self._make_service()
        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.side_effect = RuntimeError("boom")

        with patch.object(svc, "get_client", return_value=mock_client):
            result = svc(prompt="test", image=None, block=None,
                         response_schema=MagicMock(__name__="S"))

        self.assertEqual(result, {})

    def test_block_metadata_updated(self):
        svc = self._make_service()
        mock_client = MagicMock()
        mock_client.beta.chat.completions.parse.return_value = _make_openai_response({"x": 1})
        block = MagicMock()

        with patch.object(svc, "get_client", return_value=mock_client):
            svc(prompt="test", image=None, block=block,
                response_schema=MagicMock(__name__="S"))

        block.update_metadata.assert_called_once_with(llm_tokens_used=42, llm_request_count=1)


# ---------------------------------------------------------------------------
# Tests: handler routes to OpenAIServiceWithExtraBody
# ---------------------------------------------------------------------------

class TestHandlerRouting(unittest.TestCase):

    def _run(self, job_input: dict):
        return _handler_mod.handler({"input": job_input})

    def test_handler_accepts_openai_service_with_extra_body(self):
        """
        The handler should recognise openai_service.OpenAIServiceWithExtraBody
        as a valid llm_service path and complete a conversion successfully.
        """
        fake_rendered = MagicMock()
        fake_rendered.metadata = {"page_stats": [{}]}
        fake_converter = MagicMock(return_value=fake_rendered)

        fake_parser = MagicMock()
        fake_parser.generate_config_dict.return_value = {}
        fake_parser.get_processors.return_value = []
        fake_parser.get_renderer.return_value = MagicMock()
        fake_parser.get_llm_service.return_value = None

        mock_http = MagicMock()
        mock_http.content = b"%PDF-1.4 fake"
        mock_http.raise_for_status = MagicMock()

        with patch("marker.config.parser.ConfigParser", return_value=fake_parser), \
             patch("marker.converters.pdf.PdfConverter", return_value=fake_converter), \
             patch("marker.output.text_from_rendered", return_value=("# Hello", {}, {})), \
             patch("requests.get", return_value=mock_http):

            result = self._run({
                "pdf": "https://example.com/sample.pdf",
                "filename": "sample.pdf",
                "output_format": "markdown",
                "use_llm": True,
                "llm_service": "openai_service.OpenAIServiceWithExtraBody",
                "llm_config": {
                    "openai_base_url": "http://localhost:8000/v1",
                    "openai_api_key": "EMPTY",
                    "openai_model": "Qwen/Qwen2.5-VL-7B-Instruct",
                    "openai_extra_body": {
                        "chat_template_kwargs": {"enable_thinking": False},
                        "top_k": 20,
                    },
                },
            })

        self.assertTrue(result.get("success"), msg=f"Handler returned failure: {result}")
        self.assertEqual(result["output_format"], "markdown")

    def test_handler_rejects_invalid_llm_config_type(self):
        result = self._run({
            "pdf": "dGVzdA==",  # valid base64
            "filename": "test.pdf",
            "use_llm": True,
            "llm_service": "openai_service.OpenAIServiceWithExtraBody",
            "llm_config": "not-a-dict",
        })
        self.assertFalse(result.get("success"))
        self.assertIn("llm_config", result.get("error", ""))


if __name__ == "__main__":
    unittest.main()
