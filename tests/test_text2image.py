"""Tests for Text2Image support: diffusion-model detection, PNG encoding,
Backend.image_generate dispatch/kwargs, and POST /v1/images/generations.

og is never imported for assertions: pure helpers are tested directly,
dispatch tests stub _pipeline, transport tests stub the whole backend.
"""

import base64
import io
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

import numpy as np
from PIL import Image
from fastapi.testclient import TestClient

from ovinfer import config
from ovinfer import registry
from ovinfer import server as server_mod
from ovinfer.backend import Backend, ImageResult, _encode_pngs
from ovinfer.errors import ModelNotFound, OvinferError, UnsupportedModel
from ovinfer.server import create_app


def _diffusion_root(model: str = "org/diffuser") -> tuple[Path, str]:
    """Temp models_root holding a diffusion export (has model_index.json).

    The *.xml is required: list_models (used by resolve_model_dir) only counts
    a dir as a model if it holds at least one *.xml anywhere inside.
    """
    root = Path(tempfile.mkdtemp())
    d = root / model
    (d / "transformer").mkdir(parents=True)
    (d / "transformer" / "openvino_model.xml").write_text("<net/>")
    (d / "model_index.json").write_text("{}")
    return root, model


def _llm_root(model: str = "org/llm") -> tuple[Path, str]:
    """Temp models_root holding a plain-LLM export (no model_index.json)."""
    root = Path(tempfile.mkdtemp())
    d = root / model
    d.mkdir(parents=True)
    (d / "openvino_model.xml").write_text("<net/>")
    return root, model


def _decode_png(data: bytes) -> Image.Image:
    return Image.open(io.BytesIO(data))


MODEL = "org/diffuser"
_PNG = b"\x89PNG\r\n\x1a\n-fake-png-bytes"


class _StubImageBackend:
    """Duck-typed Backend for transport tests; records image_generate calls."""

    def __init__(self, result_or_exc) -> None:
        self.result_or_exc = result_or_exc
        self.calls: list[dict] = []

    def image_generate(self, model_id, prompt, *, n, width, height,
                       num_inference_steps, guidance_scale, negative_prompt, seed):
        self.calls.append({
            "model_id": model_id, "prompt": prompt, "n": n,
            "width": width, "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "negative_prompt": negative_prompt, "seed": seed,
        })
        if isinstance(self.result_or_exc, Exception):
            raise self.result_or_exc
        return self.result_or_exc


def _client(result_or_exc) -> tuple[TestClient, _StubImageBackend]:
    backend = _StubImageBackend(result_or_exc)
    return TestClient(create_app(backend)), backend


class IsDiffusionModelTest(unittest.TestCase):
    def test_diffusion_export_is_detected(self):
        root, model = _diffusion_root()
        self.assertTrue(registry.is_diffusion_model(root / model))

    def test_plain_llm_export_is_not_diffusion(self):
        root, model = _llm_root()
        self.assertFalse(registry.is_diffusion_model(root / model))

    def test_nonexistent_path_is_false(self):
        self.assertFalse(registry.is_diffusion_model(Path("/no/such/dir")))


class UnsupportedModelMappingTest(unittest.TestCase):
    def test_is_ovinfer_error(self):
        self.assertIsInstance(UnsupportedModel("x"), OvinferError)

    def test_maps_to_400_unsupported_model(self):
        status, body = server_mod._map_error(UnsupportedModel("nope"))
        self.assertEqual(status, 400)
        self.assertEqual(body["error"]["type"], "invalid_request_error")
        self.assertEqual(body["error"]["code"], "unsupported_model")
        self.assertEqual(body["error"]["message"], "nope")


class PipelineKindTest(unittest.TestCase):
    def test_text2image_is_a_known_kind(self):
        self.assertIn("text2image", config.PIPELINE_KINDS)

    def test_preload_accepts_text2image(self):
        self.assertEqual(
            config.parse_preload("org/m:text2image"), [("org/m", "text2image")]
        )


class EncodePngsTest(unittest.TestCase):
    def test_batch_tensor_becomes_one_png_per_image(self):
        arr = np.zeros((2, 4, 6, 3), dtype=np.uint8)
        arr[1, :, :, 0] = 255  # second image: red channel
        pngs = _encode_pngs(arr)
        self.assertEqual(len(pngs), 2)
        for png in pngs:
            self.assertTrue(png.startswith(b"\x89PNG"))
        # PIL size is (W, H) = (6, 4); pixel of second image is red.
        img1 = _decode_png(pngs[1])
        self.assertEqual(img1.size, (6, 4))
        self.assertEqual(img1.convert("RGB").getpixel((0, 0)), (255, 0, 0))

    def test_single_image_3d_tensor_is_normalized_to_one_png(self):
        arr = np.zeros((4, 4, 3), dtype=np.uint8)
        pngs = _encode_pngs(arr)
        self.assertEqual(len(pngs), 1)
        self.assertTrue(pngs[0].startswith(b"\x89PNG"))

    def test_image_result_holds_png_list(self):
        r = ImageResult(images=[b"\x89PNG..."])
        self.assertEqual(r.images, [b"\x89PNG..."])


class _FakeT2IPipe:
    """Stands in for og.Text2ImagePipeline: records generate kwargs, returns
    a fake (N,H,W,3) uint8 tensor sized to num_images_per_prompt."""

    def __init__(self):
        self.calls: list[dict] = []

    def generate(self, prompt, **kwargs):
        self.calls.append({"prompt": prompt, "kwargs": kwargs})
        n = kwargs.get("num_images_per_prompt", 1)
        return np.zeros((n, 4, 4, 3), dtype=np.uint8)


class _DispatchBackend(Backend):
    """Real Backend with _pipeline stubbed; records (model_id, kind) requests."""

    def __init__(self, models_root: Path, pipe: _FakeT2IPipe) -> None:
        super().__init__(models_root=models_root, default_device="CPU")
        self._pipe = pipe
        self.pipeline_requests: list[tuple[str, str]] = []

    @contextmanager
    def _pipeline(self, model_id, kind):
        self.pipeline_requests.append((model_id, kind))
        yield self._pipe


class ImageGenerateTest(unittest.TestCase):
    def test_requests_text2image_kind_and_returns_pngs(self):
        root, model = _diffusion_root()
        pipe = _FakeT2IPipe()
        be = _DispatchBackend(root, pipe)
        result = be.image_generate(model, "a cat", n=2)
        self.assertIsInstance(result, ImageResult)
        self.assertEqual(len(result.images), 2)
        self.assertTrue(result.images[0].startswith(b"\x89PNG"))
        self.assertEqual(be.pipeline_requests, [(model, "text2image")])

    def test_only_set_params_reach_generate(self):
        root, model = _diffusion_root()
        pipe = _FakeT2IPipe()
        be = _DispatchBackend(root, pipe)
        be.image_generate(model, "x")
        # Omitted optionals must not appear (so og defaults are not clobbered).
        self.assertEqual(pipe.calls[0]["kwargs"], {"num_images_per_prompt": 1})
        self.assertEqual(pipe.calls[0]["prompt"], "x")

    def test_all_params_map_to_og_kwargs(self):
        root, model = _diffusion_root()
        pipe = _FakeT2IPipe()
        be = _DispatchBackend(root, pipe)
        be.image_generate(
            model, "x", n=3, width=512, height=768,
            num_inference_steps=8, guidance_scale=0.0,
            negative_prompt="blurry", seed=7,
        )
        self.assertEqual(pipe.calls[0]["kwargs"], {
            "num_images_per_prompt": 3,
            "width": 512,
            "height": 768,
            "num_inference_steps": 8,
            "guidance_scale": 0.0,   # 0.0 must pass through, not be dropped
            "negative_prompt": "blurry",
            "rng_seed": 7,
        })

    def test_non_diffusion_model_is_unsupported_without_building_pipeline(self):
        root, model = _llm_root()
        be = _DispatchBackend(root, _FakeT2IPipe())
        with self.assertRaises(UnsupportedModel):
            be.image_generate(model, "x")
        self.assertEqual(be.pipeline_requests, [])

    def test_unknown_model_404s_before_any_pipeline(self):
        root, _ = _diffusion_root()
        be = _DispatchBackend(root, _FakeT2IPipe())
        with self.assertRaises(ModelNotFound):
            be.image_generate("org/ghost", "x")
        self.assertEqual(be.pipeline_requests, [])


class ParseSizeTest(unittest.TestCase):
    def test_width_x_height(self):
        self.assertEqual(server_mod._parse_size("1024x768"), (1024, 768))

    def test_square(self):
        self.assertEqual(server_mod._parse_size("512x512"), (512, 512))

    def test_none_means_model_default(self):
        self.assertEqual(server_mod._parse_size(None), (None, None))

    def test_auto_means_model_default(self):
        self.assertEqual(server_mod._parse_size("auto"), (None, None))

    def test_missing_separator_raises(self):
        with self.assertRaises(ValueError):
            server_mod._parse_size("1024")

    def test_non_integer_raises(self):
        with self.assertRaises(ValueError):
            server_mod._parse_size("axb")


class ImagesGenerationRouteTest(unittest.TestCase):
    def _post(self, client, **overrides):
        body = {"model": MODEL, "prompt": "a cat"}
        body.update(overrides)
        return client.post("/v1/images/generations", json=body)

    def test_b64_json_response_shape(self):
        client, _ = _client(ImageResult(images=[_PNG]))
        resp = self._post(client)
        self.assertEqual(resp.status_code, 200)
        data = resp.json()
        self.assertIsInstance(data["created"], int)
        self.assertEqual(len(data["data"]), 1)
        self.assertEqual(
            data["data"][0]["b64_json"], base64.b64encode(_PNG).decode("ascii")
        )

    def test_url_format_returns_data_url(self):
        client, _ = _client(ImageResult(images=[_PNG]))
        resp = self._post(client, response_format="url")
        entry = resp.json()["data"][0]
        self.assertTrue(entry["url"].startswith("data:image/png;base64,"))
        self.assertEqual(
            entry["url"],
            "data:image/png;base64," + base64.b64encode(_PNG).decode("ascii"),
        )

    def test_extension_fields_forwarded_with_parsed_size(self):
        client, backend = _client(ImageResult(images=[_PNG]))
        self._post(
            client, n=2, size="512x768", num_inference_steps=8,
            guidance_scale=0.0, negative_prompt="blurry", seed=7,
        )
        self.assertEqual(backend.calls, [{
            "model_id": MODEL, "prompt": "a cat", "n": 2,
            "width": 512, "height": 768,
            "num_inference_steps": 8, "guidance_scale": 0.0,
            "negative_prompt": "blurry", "seed": 7,
        }])

    def test_omitted_optionals_forwarded_as_none(self):
        client, backend = _client(ImageResult(images=[_PNG]))
        self._post(client)
        self.assertEqual(backend.calls[0], {
            "model_id": MODEL, "prompt": "a cat", "n": 1,
            "width": None, "height": None,
            "num_inference_steps": None, "guidance_scale": None,
            "negative_prompt": None, "seed": None,
        })

    def test_bad_size_is_400_without_calling_backend(self):
        client, backend = _client(ImageResult(images=[_PNG]))
        resp = self._post(client, size="1024")
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["error"]["code"], "invalid_size")
        self.assertEqual(backend.calls, [])

    def test_unsupported_model_maps_to_400(self):
        client, _ = _client(UnsupportedModel("not a diffusion model"))
        resp = self._post(client)
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["error"]["code"], "unsupported_model")

    def test_model_not_found_maps_to_404(self):
        client, _ = _client(ModelNotFound("org/ghost", Path("/nowhere")))
        resp = self._post(client, model="org/ghost")
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.json()["error"]["code"], "model_not_found")


if __name__ == "__main__":
    unittest.main()
