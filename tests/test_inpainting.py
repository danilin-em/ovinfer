"""Tests for Inpainting support: the inpainting pipeline kind, the hybrid mask
decoder, Backend.image_edit's mask branch (dispatch/kwargs/mask-routing), and
POST /v1/images/edits with a mask (multipart).

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
    """A small valid RGB PNG. PIL size is (W, H); tensor shape is (H, W, 3)."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGB", size, color).save(buf, format="PNG")
    return buf.getvalue()


def _mask_bytes(size=(6, 4)) -> bytes:
    """A valid RGBA mask PNG; content is irrelevant for dispatch/transport tests
    (the fake pipe ignores it). Decode-semantics tests build arrays inline."""
    from PIL import Image

    buf = io.BytesIO()
    Image.new("RGBA", size, (0, 0, 0, 0)).save(buf, format="PNG")
    return buf.getvalue()


class PipelineKindTest(unittest.TestCase):
    def test_inpainting_is_a_known_kind(self):
        self.assertIn("inpainting", config.PIPELINE_KINDS)

    def test_preload_accepts_inpainting(self):
        self.assertEqual(
            config.parse_preload("org/m:inpainting"), [("org/m", "inpainting")]
        )


import numpy as np

from ovinfer.backend import _decode_mask_tensor
from ovinfer.errors import InvalidImage


class DecodeMaskTensorTest(unittest.TestCase):
    def test_rgba_transparent_region_becomes_white_edit(self):
        from PIL import Image

        # 4x4 RGBA: top-left 2x2 transparent (alpha 0 -> edit), rest opaque.
        arr = np.zeros((4, 4, 4), dtype=np.uint8)
        arr[..., 3] = 255            # opaque everywhere (keep)
        arr[0:2, 0:2, 3] = 0         # top-left transparent (edit)
        buf = io.BytesIO()
        Image.fromarray(arr, "RGBA").save(buf, format="PNG")

        out = np.asarray(_decode_mask_tensor(buf.getvalue()).data)
        self.assertEqual(list(out.shape), [1, 4, 4, 3])
        # transparent -> white (repaint); opaque -> black (keep).
        self.assertTrue((out[0, 0:2, 0:2, :] == 255).all())
        self.assertTrue((out[0, 2:, 2:, :] == 0).all())

    def test_grayscale_mask_white_marks_repaint(self):
        from PIL import Image

        # 4x4 L: top-left 2x2 white (repaint), rest black (keep). No alpha ->
        # luminance is used as-is (white -> repaint), not inverted.
        arr = np.zeros((4, 4), dtype=np.uint8)
        arr[0:2, 0:2] = 255
        buf = io.BytesIO()
        Image.fromarray(arr, "L").save(buf, format="PNG")

        out = np.asarray(_decode_mask_tensor(buf.getvalue()).data)
        self.assertEqual(list(out.shape), [1, 4, 4, 3])
        self.assertTrue((out[0, 0:2, 0:2, :] == 255).all())
        self.assertTrue((out[0, 2:, 2:, :] == 0).all())

    def test_undecodable_mask_raises_invalid_image(self):
        with self.assertRaises(InvalidImage):
            _decode_mask_tensor(b"not an image")


if __name__ == "__main__":
    unittest.main()
