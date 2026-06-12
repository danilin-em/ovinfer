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


from ovinfer.backend import Backend, ImageResult
from ovinfer.errors import ModelNotFound, UnsupportedModel


class _FakeInpaintPipe:
    """Stands in for og.InpaintingPipeline: records the prompt, the image and
    mask tensors (positional), and generate kwargs; returns a fake (N,H,W,3)
    uint8 tensor sized to num_images_per_prompt."""

    def __init__(self):
        self.calls: list[dict] = []

    def generate(self, prompt, image, mask_image, **kwargs):
        self.calls.append(
            {"prompt": prompt, "image": image, "mask": mask_image, "kwargs": kwargs}
        )
        n = kwargs.get("num_images_per_prompt", 1)
        return np.zeros((n, 4, 4, 3), dtype=np.uint8)


class _DispatchBackend(Backend):
    """Real Backend with _pipeline stubbed; records (model_id, kind) requests."""

    def __init__(self, models_root: Path, pipe: _FakeInpaintPipe) -> None:
        super().__init__(models_root=models_root, default_device="CPU")
        self._pipe = pipe
        self.pipeline_requests: list[tuple[str, str]] = []

    @contextmanager
    def _pipeline(self, model_id, kind):
        self.pipeline_requests.append((model_id, kind))
        yield self._pipe


class InpaintEditDispatchTest(unittest.TestCase):
    def test_mask_requests_inpainting_kind_and_forwards_image_and_mask(self):
        root, model = _diffusion_root()
        pipe = _FakeInpaintPipe()
        be = _DispatchBackend(root, pipe)
        result = be.image_edit(
            model, "a cat", _png_bytes(size=(6, 4)), _mask_bytes(size=(6, 4)), n=2
        )
        self.assertIsInstance(result, ImageResult)
        self.assertEqual(len(result.images), 2)
        self.assertTrue(result.images[0].startswith(b"\x89PNG"))
        self.assertEqual(be.pipeline_requests, [(model, "inpainting")])
        call = pipe.calls[0]
        self.assertEqual(call["prompt"], "a cat")
        # Both tensors are batched NHWC (1, H, W, 3) = (1, 4, 6, 3).
        self.assertEqual(list(call["image"].shape), [1, 4, 6, 3])
        self.assertEqual(list(call["mask"].shape), [1, 4, 6, 3])

    def test_only_set_params_reach_generate(self):
        root, model = _diffusion_root()
        pipe = _FakeInpaintPipe()
        be = _DispatchBackend(root, pipe)
        be.image_edit(model, "x", _png_bytes(), _mask_bytes())
        # Omitted optionals must not appear (so og defaults, incl. strength=0.8,
        # are not clobbered).
        self.assertEqual(pipe.calls[0]["kwargs"], {"num_images_per_prompt": 1})

    def test_all_params_map_to_og_kwargs(self):
        root, model = _diffusion_root()
        pipe = _FakeInpaintPipe()
        be = _DispatchBackend(root, pipe)
        be.image_edit(
            model, "x", _png_bytes(), _mask_bytes(), n=3, width=512, height=768,
            num_inference_steps=8, guidance_scale=0.0,
            negative_prompt="blurry", strength=0.0, seed=7,
        )
        self.assertEqual(pipe.calls[0]["kwargs"], {
            "num_images_per_prompt": 3,
            "width": 512,
            "height": 768,
            "num_inference_steps": 8,
            "guidance_scale": 0.0,   # 0.0 must pass through, not be dropped
            "negative_prompt": "blurry",
            "strength": 0.0,         # 0.0 must pass through, not be dropped
            "rng_seed": 7,
        })

    def test_non_diffusion_model_with_mask_is_unsupported_without_pipeline(self):
        # The diffusion check now precedes the mask split: a mask on a
        # non-diffusion model is 400 (not 501).
        root, model = _llm_root()
        be = _DispatchBackend(root, _FakeInpaintPipe())
        with self.assertRaises(UnsupportedModel):
            be.image_edit(model, "x", _png_bytes(), _mask_bytes())
        self.assertEqual(be.pipeline_requests, [])

    def test_undecodable_mask_is_invalid_image_without_pipeline(self):
        root, model = _diffusion_root()
        be = _DispatchBackend(root, _FakeInpaintPipe())
        with self.assertRaises(InvalidImage):
            be.image_edit(model, "x", _png_bytes(), b"not an image")
        self.assertEqual(be.pipeline_requests, [])

    def test_undecodable_image_with_mask_is_invalid_image_without_pipeline(self):
        # Image is decoded before the mask, so a broken image 400s first.
        root, model = _diffusion_root()
        be = _DispatchBackend(root, _FakeInpaintPipe())
        with self.assertRaises(InvalidImage):
            be.image_edit(model, "x", b"not an image", _mask_bytes())
        self.assertEqual(be.pipeline_requests, [])

    def test_unknown_model_with_mask_404s_before_anything(self):
        root, _ = _diffusion_root()
        be = _DispatchBackend(root, _FakeInpaintPipe())
        with self.assertRaises(ModelNotFound):
            be.image_edit("org/ghost", "x", _png_bytes(), _mask_bytes())
        self.assertEqual(be.pipeline_requests, [])


if __name__ == "__main__":
    unittest.main()
