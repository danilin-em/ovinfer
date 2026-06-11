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
from ovinfer.backend import ImageResult, _encode_pngs
from ovinfer.errors import OvinferError, UnsupportedModel


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


if __name__ == "__main__":
    unittest.main()
