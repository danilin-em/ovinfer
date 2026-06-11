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

from ovinfer import config
from ovinfer import registry
from ovinfer import server as server_mod
from ovinfer.backend import Backend, ImageResult, _encode_pngs
from ovinfer.errors import ModelNotFound, OvinferError, UnsupportedModel


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


if __name__ == "__main__":
    unittest.main()
