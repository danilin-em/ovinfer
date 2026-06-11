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

from ovinfer import backend as backend_mod
from ovinfer.backend import (
    Backend,
    ChatResult,
    _extract_image_url,
    _fetch_image_bytes,
    _vlm_history,
)
from ovinfer.errors import InvalidImage, ModelNotFound
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


def _llm_root(model: str = "org/llm") -> tuple[Path, str]:
    """Temp models_root holding a plain-LLM export (no vision IR).

    The *.xml file is required: list_models (used by resolve_model_dir) only
    counts a dir as a model if it holds at least one *.xml — do not remove it.
    """
    root = Path(tempfile.mkdtemp())
    d = root / model
    d.mkdir(parents=True)
    (d / "openvino_model.xml").write_text("<net/>")
    return root, model


def _vlm_root(model: str = "org/vlm") -> tuple[Path, str]:
    """Temp models_root holding a VLM export (has a vision-encoder IR)."""
    root = Path(tempfile.mkdtemp())
    d = root / model
    d.mkdir(parents=True)
    (d / "openvino_language_model.xml").write_text("<net/>")
    (d / "openvino_vision_embeddings_model.xml").write_text("<net/>")
    return root, model


class IsVlmModelTest(unittest.TestCase):
    def test_vlm_export_is_detected(self):
        from ovinfer import registry

        root, model = _vlm_root()
        self.assertTrue(registry.is_vlm_model(root / model))

    def test_plain_llm_export_is_not_vlm(self):
        from ovinfer import registry

        root, model = _llm_root()
        self.assertFalse(registry.is_vlm_model(root / model))


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


class _FakeMetrics:
    def __init__(self, prompt_tokens: int, completion_tokens: int) -> None:
        self._p, self._c = prompt_tokens, completion_tokens

    def get_num_input_tokens(self) -> int:
        return self._p

    def get_num_generated_tokens(self) -> int:
        return self._c


class _FakeResult:
    def __init__(self, text: str = "reply", prompt_tokens: int = 3,
                 completion_tokens: int = 2) -> None:
        self.texts = [text]
        self.perf_metrics = _FakeMetrics(prompt_tokens, completion_tokens)


class _FakeTokenizer:
    def apply_chat_template(self, history, add_generation_prompt):
        # Echo a deterministic rendering so tests can assert on it.
        return "TPL:" + "|".join(f"{m['role']}={m['content']}" for m in history)


class _FakeConfig:
    def __init__(self) -> None:
        self.max_new_tokens = 512
        self.apply_chat_template = True


class _FakeChatPipe:
    """Stands in for both og.LLMPipeline and og.VLMPipeline."""

    def __init__(self, deltas: list[str] | None = None) -> None:
        self.generate_calls: list[dict] = []
        self._deltas = deltas or []

    def get_tokenizer(self):
        return _FakeTokenizer()

    def generate(self, prompt, *args, images=None, generation_config=None,
                 streamer=None, **kwargs):
        # LLM call:  generate([prompt], cfg[, on_token])  -> cfg in args[0]
        # VLM call:  generate(prompt, images=..., generation_config=cfg, streamer=..)
        cfg = generation_config if generation_config is not None else (
            args[0] if args else None
        )
        on_token = streamer if streamer is not None else (
            args[1] if len(args) > 1 else None
        )
        self.generate_calls.append(
            {"prompt": prompt, "images": images, "cfg": cfg}
        )
        if on_token is not None:
            for d in self._deltas:
                on_token(d)
        return _FakeResult()


class _DispatchBackend(Backend):
    """Real Backend with _pipeline and _generation_config stubbed."""

    def __init__(self, models_root: Path, pipe: _FakeChatPipe) -> None:
        super().__init__(models_root=models_root, default_device="CPU")
        self._pipe = pipe
        self.pipeline_requests: list[tuple[str, str]] = []

    @contextmanager
    def _pipeline(self, model_id, kind):
        self.pipeline_requests.append((model_id, kind))
        yield self._pipe

    @staticmethod
    def _generation_config(temperature, top_p, max_tokens):
        return _FakeConfig()


class DispatchTest(unittest.TestCase):
    def test_text_only_to_llm_model_uses_llm_pipeline(self):
        root, model = _llm_root()
        pipe = _FakeChatPipe()
        be = _DispatchBackend(root, pipe)
        result = be.chat(model, [{"role": "user", "content": "hi"}])
        self.assertIsInstance(result, ChatResult)
        self.assertEqual(be.pipeline_requests, [(model, "llm")])
        # LLM path passes a 1-element list and no images.
        self.assertEqual(pipe.generate_calls[0]["prompt"], ["TPL:user=hi"])
        self.assertIsNone(pipe.generate_calls[0]["images"])

    def test_text_only_to_vlm_model_uses_vlm_pipeline(self):
        root, model = _vlm_root()
        pipe = _FakeChatPipe()
        be = _DispatchBackend(root, pipe)
        be.chat(model, [{"role": "user", "content": "hi"}])
        self.assertEqual(be.pipeline_requests, [(model, "vlm")])
        call = pipe.generate_calls[0]
        self.assertEqual(call["prompt"], "TPL:user=hi")  # plain str, not a list
        self.assertEqual(call["images"], [])             # empty, not None
        self.assertFalse(call["cfg"].apply_chat_template)  # forced False for VLM

    def test_image_to_llm_model_is_invalid_image_without_building_pipeline(self):
        root, model = _llm_root()
        be = _DispatchBackend(root, _FakeChatPipe())
        msg = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "http://x/a.png"}},
                {"type": "text", "text": "?"},
            ],
        }
        with self.assertRaises(InvalidImage):
            be.chat(model, [msg])
        self.assertEqual(be.pipeline_requests, [])

    def test_image_to_vlm_model_loads_and_passes_tensors(self):
        root, model = _vlm_root()
        pipe = _FakeChatPipe()
        be = _DispatchBackend(root, pipe)
        msg = {
            "role": "user",
            "content": [
                {"type": "image_url", "image_url": {"url": "u0"}},
                {"type": "text", "text": "describe"},
            ],
        }
        with mock.patch.object(backend_mod, "_load_image_tensor", return_value="T0"):
            be.chat(model, [msg])
        call = pipe.generate_calls[0]
        self.assertEqual(call["images"], ["T0"])
        self.assertEqual(call["prompt"], "TPL:user=<ov_genai_image_0>describe")

    def test_unknown_model_404s_before_any_image_load(self):
        root, _ = _vlm_root()
        be = _DispatchBackend(root, _FakeChatPipe())
        with self.assertRaises(ModelNotFound):
            be.chat("org/ghost", [{"role": "user", "content": "hi"}])


class StreamDispatchTest(unittest.TestCase):
    @staticmethod
    def _drain(gen):
        """Collect yielded deltas and the generator's return value (the
        finish_reason carried by StopIteration.value, PEP 380)."""
        deltas = []
        try:
            while True:
                deltas.append(next(gen))
        except StopIteration as stop:
            return deltas, stop.value

    def test_vlm_stream_uses_vlm_pipeline_and_yields_deltas(self):
        root, model = _vlm_root()
        pipe = _FakeChatPipe(deltas=["Hello", " world"])
        be = _DispatchBackend(root, pipe)
        gen = be.chat_stream(model, [{"role": "user", "content": "hi"}])
        deltas, finish_reason = self._drain(gen)
        self.assertEqual(deltas, ["Hello", " world"])
        # _FakeResult reports 2 generated tokens vs max_new_tokens=512 -> "stop";
        # confirms _chat_stream_gen propagates finish_reason as its return value.
        self.assertEqual(finish_reason, "stop")
        self.assertEqual(be.pipeline_requests, [(model, "vlm")])

    def test_llm_stream_uses_llm_pipeline(self):
        root, model = _llm_root()
        pipe = _FakeChatPipe(deltas=["a", "b"])
        be = _DispatchBackend(root, pipe)
        gen = be.chat_stream(model, [{"role": "user", "content": "hi"}])
        self.assertEqual(list(gen), ["a", "b"])
        self.assertEqual(be.pipeline_requests, [(model, "llm")])


if __name__ == "__main__":
    unittest.main()
