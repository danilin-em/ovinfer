"""Tests for Image2Image support: the image2image pipeline kind, the
byte-level image decoder, Backend.image_edit dispatch/kwargs/mask-routing,
and POST /v1/images/edits (multipart).

og (openvino_genai) is never imported for assertions: pure helpers are tested
directly, dispatch tests stub _pipeline, transport tests stub the whole backend.
openvino/PIL/numpy are core deps and may be imported.
"""

import base64
import io
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from ovinfer import config


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


def _png_bytes(color=(80, 120, 200), size=(6, 4)) -> bytes:
    """A small valid PNG. PIL size is (W, H); numpy/tensor shape is (H, W, 3)."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


class PipelineKindTest(unittest.TestCase):
    def test_image2image_is_a_known_kind(self):
        self.assertIn("image2image", config.PIPELINE_KINDS)

    def test_preload_accepts_image2image(self):
        self.assertEqual(
            config.parse_preload("org/m:image2image"), [("org/m", "image2image")]
        )


from ovinfer.backend import _decode_image_tensor, _load_image_tensor
from ovinfer.errors import InvalidImage


class DecodeImageTensorTest(unittest.TestCase):
    def test_valid_png_bytes_become_rgb_tensor(self):
        tensor = _decode_image_tensor(_png_bytes(size=(3, 2)))
        # PIL image is (W=3, H=2); tensor is (H, W, C) = (2, 3, 3).
        self.assertEqual(list(tensor.shape), [2, 3, 3])

    def test_undecodable_bytes_raise_invalid_image(self):
        with self.assertRaises(InvalidImage):
            _decode_image_tensor(b"not an image")

    def test_load_image_tensor_still_decodes_a_data_url(self):
        # Regression: the VLM entry point must keep working after the refactor.
        url = "data:image/png;base64," + base64.b64encode(_png_bytes(size=(3, 2))).decode()
        tensor = _load_image_tensor(url)
        self.assertEqual(list(tensor.shape), [2, 3, 3])


if __name__ == "__main__":
    unittest.main()
