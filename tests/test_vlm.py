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

from ovinfer.backend import (
    ChatResult,
    _extract_image_url,
    _fetch_image_bytes,
    _vlm_history,
)
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


class VlmHistoryTest(unittest.TestCase):
    def test_plain_string_content_passes_through(self):
        history, urls = _vlm_history([{"role": "user", "content": "hi"}])
        self.assertEqual(history, [{"role": "user", "content": "hi"}])
        self.assertEqual(urls, [])

    def test_none_content_renders_empty_string(self):
        history, urls = _vlm_history([{"role": "assistant", "content": None}])
        self.assertEqual(history, [{"role": "assistant", "content": ""}])
        self.assertEqual(urls, [])

    def test_text_kept_image_becomes_tag_in_part_order(self):
        msg = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "http://x/a.png"}},
                {"type": "text", "text": "what is this?"},
            ],
        }
        history, urls = _vlm_history([msg])
        self.assertEqual(
            history, [{"role": "user", "content": "<ov_genai_image_0>what is this?"}]
        )
        self.assertEqual(urls, ["http://x/a.png"])

    def test_running_index_across_messages(self):
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image_url", "image_url": {"url": "u0"}},
                    {"type": "text", "text": "first"},
                ],
            },
            {"role": "assistant", "content": "ok"},
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "and "},
                    {"type": "input_image", "image_url": "u1"},
                ],
            },
        ]
        history, urls = _vlm_history(messages)
        self.assertEqual(urls, ["u0", "u1"])
        self.assertEqual(history[0]["content"], "<ov_genai_image_0>first")
        self.assertEqual(history[1], {"role": "assistant", "content": "ok"})
        self.assertEqual(history[2]["content"], "and <ov_genai_image_1>")


class ExtractImageUrlTest(unittest.TestCase):
    def test_dict_form(self):
        self.assertEqual(
            _extract_image_url({"type": "image_url", "image_url": {"url": "u"}}), "u"
        )

    def test_string_form(self):
        self.assertEqual(
            _extract_image_url({"type": "input_image", "image_url": "u"}), "u"
        )

    def test_missing_url_raises(self):
        with self.assertRaises(InvalidImage):
            _extract_image_url({"type": "image_url", "image_url": {}})
        with self.assertRaises(InvalidImage):
            _extract_image_url({"type": "image_url"})


class FetchImageBytesTest(unittest.TestCase):
    def test_base64_data_url(self):
        url = "data:image/png;base64," + base64.b64encode(b"hello").decode()
        self.assertEqual(_fetch_image_bytes(url), b"hello")

    def test_malformed_data_url_raises(self):
        with self.assertRaises(InvalidImage):
            _fetch_image_bytes("data:nocomma")

    def test_bad_base64_raises(self):
        with self.assertRaises(InvalidImage):
            _fetch_image_bytes("data:image/png;base64,!!!notbase64!!!")

    def test_percent_encoded_data_url(self):
        # A data: URL without ;base64 carries percent-encoded bytes.
        self.assertEqual(_fetch_image_bytes("data:text/plain,hi%21"), b"hi!")

    def test_unsupported_scheme_raises(self):
        with self.assertRaises(InvalidImage):
            _fetch_image_bytes("ftp://host/img.png")


def _png_data_url(color=(255, 0, 0), size=(2, 2)) -> str:
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


class LoadImageTensorTest(unittest.TestCase):
    def test_valid_png_data_url_becomes_rgb_tensor(self):
        from ovinfer.backend import _load_image_tensor

        tensor = _load_image_tensor(_png_data_url(size=(3, 2)))
        # PIL image is (W=3, H=2); numpy array is (H, W, C) = (2, 3, 3).
        self.assertEqual(list(tensor.shape), [2, 3, 3])

    def test_undecodable_bytes_raise_invalid_image(self):
        from ovinfer.backend import _load_image_tensor

        garbage = "data:image/png;base64," + base64.b64encode(b"not an image").decode()
        with self.assertRaises(InvalidImage):
            _load_image_tensor(garbage)


if __name__ == "__main__":
    unittest.main()
