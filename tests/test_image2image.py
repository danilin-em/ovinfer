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


import numpy as np

from ovinfer.backend import Backend, ImageResult
from ovinfer.errors import ModelNotFound, PipelineNotImplemented, UnsupportedModel


class _FakeI2IPipe:
    """Stands in for og.Image2ImagePipeline: records the prompt, the image
    tensor (positional), and generate kwargs; returns a fake (N,H,W,3) uint8
    tensor sized to num_images_per_prompt."""

    def __init__(self):
        self.calls: list[dict] = []

    def generate(self, prompt, image, **kwargs):
        self.calls.append({"prompt": prompt, "image": image, "kwargs": kwargs})
        n = kwargs.get("num_images_per_prompt", 1)
        return np.zeros((n, 4, 4, 3), dtype=np.uint8)


class _DispatchBackend(Backend):
    """Real Backend with _pipeline stubbed; records (model_id, kind) requests."""

    def __init__(self, models_root: Path, pipe: _FakeI2IPipe) -> None:
        super().__init__(models_root=models_root, default_device="CPU")
        self._pipe = pipe
        self.pipeline_requests: list[tuple[str, str]] = []

    @contextmanager
    def _pipeline(self, model_id, kind):
        self.pipeline_requests.append((model_id, kind))
        yield self._pipe


class ImageEditDispatchTest(unittest.TestCase):
    def test_requests_image2image_kind_passes_image_and_returns_pngs(self):
        root, model = _diffusion_root()
        pipe = _FakeI2IPipe()
        be = _DispatchBackend(root, pipe)
        result = be.image_edit(model, "a cat", _png_bytes(size=(6, 4)), None, n=2)
        self.assertIsInstance(result, ImageResult)
        self.assertEqual(len(result.images), 2)
        self.assertTrue(result.images[0].startswith(b"\x89PNG"))
        self.assertEqual(be.pipeline_requests, [(model, "image2image")])
        # The decoded image tensor is forwarded as the second positional arg,
        # shape (H, W, 3) = (4, 6, 3).
        self.assertEqual(list(pipe.calls[0]["image"].shape), [4, 6, 3])
        self.assertEqual(pipe.calls[0]["prompt"], "a cat")

    def test_only_set_params_reach_generate(self):
        root, model = _diffusion_root()
        pipe = _FakeI2IPipe()
        be = _DispatchBackend(root, pipe)
        be.image_edit(model, "x", _png_bytes(), None)
        # Omitted optionals must not appear (so og defaults, incl. strength=0.8,
        # are not clobbered).
        self.assertEqual(pipe.calls[0]["kwargs"], {"num_images_per_prompt": 1})

    def test_all_params_map_to_og_kwargs(self):
        root, model = _diffusion_root()
        pipe = _FakeI2IPipe()
        be = _DispatchBackend(root, pipe)
        be.image_edit(
            model, "x", _png_bytes(), None, n=3, width=512, height=768,
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

    def test_empty_mask_is_treated_as_no_mask(self):
        root, model = _diffusion_root()
        pipe = _FakeI2IPipe()
        be = _DispatchBackend(root, pipe)
        be.image_edit(model, "x", _png_bytes(), b"")  # zero-length mask upload
        self.assertEqual(be.pipeline_requests, [(model, "image2image")])

    def test_non_empty_mask_is_501_inpainting_without_building_pipeline(self):
        root, model = _diffusion_root()
        be = _DispatchBackend(root, _FakeI2IPipe())
        with self.assertRaises(PipelineNotImplemented) as ctx:
            be.image_edit(model, "x", _png_bytes(), b"\x89PNG-mask")
        self.assertEqual(ctx.exception.pipeline, "InpaintingPipeline")
        self.assertEqual(be.pipeline_requests, [])

    def test_non_diffusion_model_is_unsupported_without_building_pipeline(self):
        root, model = _llm_root()
        be = _DispatchBackend(root, _FakeI2IPipe())
        with self.assertRaises(UnsupportedModel):
            be.image_edit(model, "x", _png_bytes(), None)
        self.assertEqual(be.pipeline_requests, [])

    def test_undecodable_image_is_invalid_image_without_building_pipeline(self):
        root, model = _diffusion_root()
        be = _DispatchBackend(root, _FakeI2IPipe())
        with self.assertRaises(InvalidImage):
            be.image_edit(model, "x", b"not an image", None)
        self.assertEqual(be.pipeline_requests, [])

    def test_unknown_model_404s_before_anything_else(self):
        root, _ = _diffusion_root()
        be = _DispatchBackend(root, _FakeI2IPipe())
        # Non-empty mask too: model resolution must win the 404 over the mask 501.
        with self.assertRaises(ModelNotFound):
            be.image_edit("org/ghost", "x", _png_bytes(), b"mask")
        self.assertEqual(be.pipeline_requests, [])


if __name__ == "__main__":
    unittest.main()
