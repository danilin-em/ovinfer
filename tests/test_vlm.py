"""Tests for VLM support: image-parsing helpers, chat dispatch (text-only vs
VLM vs images-to-LLM), and the InvalidImage -> 400 transport mapping.

og is never imported for assertions: helper tests are pure, dispatch tests
stub _pipeline and _generation_config, transport tests stub the whole backend.
"""

import base64
import io
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path
from unittest import mock

from fastapi.testclient import TestClient

from ovinfer.backend import ChatResult
from ovinfer.errors import InvalidImage
from ovinfer.server import create_app


class _StubBackend:
    """Duck-typed Backend for transport tests; chat/chat_stream raise or return."""

    def __init__(self, exc: Exception | None = None) -> None:
        self.exc = exc

    def chat(self, model_id, messages, **kwargs) -> ChatResult:
        if self.exc:
            raise self.exc
        return ChatResult(
            text="ok", finish_reason="stop", prompt_tokens=1, completion_tokens=1
        )

    def chat_stream(self, model_id, messages, **kwargs):
        if self.exc:
            raise self.exc

        def gen():
            yield "ok"
            return "stop"

        return gen()


_IMAGE_MSG = [
    {
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": "http://x/a.png"}}],
    }
]


class TransportInvalidImageTest(unittest.TestCase):
    def _client(self, exc):
        return TestClient(create_app(_StubBackend(exc)))

    def test_invalid_image_maps_to_400(self):
        resp = self._client(InvalidImage("bad image")).post(
            "/v1/chat/completions", json={"model": "org/m", "messages": _IMAGE_MSG}
        )
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["error"]["code"], "invalid_image")
        self.assertEqual(resp.json()["error"]["type"], "invalid_request_error")

    def test_invalid_image_in_stream_maps_to_400(self):
        # chat_stream raises eagerly (before StreamingResponse), so the
        # exception handler still produces a JSON 400, not an SSE event.
        resp = self._client(InvalidImage("bad image")).post(
            "/v1/chat/completions",
            json={"model": "org/m", "stream": True, "messages": _IMAGE_MSG},
        )
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["error"]["code"], "invalid_image")


if __name__ == "__main__":
    unittest.main()
