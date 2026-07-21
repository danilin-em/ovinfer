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


from ovinfer.backend import _decode_image_tensor, _load_image_tensor
from ovinfer.errors import InvalidImage


class DecodeImageTensorTest(unittest.TestCase):
    def test_valid_png_bytes_become_rgb_tensor(self):
        tensor = _decode_image_tensor(_png_bytes(size=(3, 2)))
        # PIL image is (W=3, H=2); tensor is (H, W, C) = (2, 3, 3).
        self.assertEqual(list(tensor.shape), [2, 3, 3])

    def test_batched_adds_a_leading_batch_dim(self):
        # og.Image2ImagePipeline wants a 4D NHWC (1,H,W,3) tensor, not bare HWC.
        tensor = _decode_image_tensor(_png_bytes(size=(3, 2)), batched=True)
        self.assertEqual(list(tensor.shape), [1, 2, 3, 3])

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
from ovinfer.errors import ModelNotFound, UnsupportedModel


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
        # batched for og: NHWC (1, H, W, 3) = (1, 4, 6, 3).
        self.assertEqual(list(pipe.calls[0]["image"].shape), [1, 4, 6, 3])
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
        # Non-empty mask too: model resolution must win the 404 over the
        # mask/diffusion checks.
        with self.assertRaises(ModelNotFound):
            be.image_edit("org/ghost", "x", _png_bytes(), b"mask")
        self.assertEqual(be.pipeline_requests, [])

    def test_non_diffusion_model_is_unsupported_even_with_mask(self):
        # The diffusion check precedes the mask split: a mask on a non-diffusion
        # model is 400 UnsupportedModel (Inpainting needs a diffusion model too),
        # not 501.
        root, model = _llm_root()
        be = _DispatchBackend(root, _FakeI2IPipe())
        with self.assertRaises(UnsupportedModel):
            be.image_edit(model, "x", _png_bytes(), b"mask")
        self.assertEqual(be.pipeline_requests, [])


from fastapi.testclient import TestClient

from ovinfer.server import create_app

MODEL = "org/diffuser"
_PNG = b"\x89PNG\r\n\x1a\n-fake-png-bytes"


class _StubEditBackend:
    """Duck-typed Backend for transport tests; records image_edit calls.

    The signature mirrors the real Backend.image_edit: image/mask positional,
    the diffusion knobs keyword-only.
    """

    def __init__(self, result_or_exc) -> None:
        self.result_or_exc = result_or_exc
        self.calls: list[dict] = []

    def image_edit(self, model_id, prompt, image, mask, *, n, width, height,
                   num_inference_steps, guidance_scale, negative_prompt,
                   strength, seed):
        self.calls.append({
            "model_id": model_id, "prompt": prompt, "image": image, "mask": mask,
            "n": n, "width": width, "height": height,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale, "negative_prompt": negative_prompt,
            "strength": strength, "seed": seed,
        })
        if isinstance(self.result_or_exc, Exception):
            raise self.result_or_exc
        return self.result_or_exc


def _edit_client(result_or_exc) -> tuple[TestClient, _StubEditBackend]:
    backend = _StubEditBackend(result_or_exc)
    return TestClient(create_app(backend)), backend


class ImagesEditsRouteTest(unittest.TestCase):
    def _post(self, client, *, data=None, files=None):
        form = {"model": MODEL, "prompt": "a cat"}
        if data:
            form.update(data)
        upload = files if files is not None else {
            "image": ("in.png", _png_bytes(), "image/png")
        }
        return client.post("/v1/images/edits", data=form, files=upload)

    def test_b64_json_response_shape_and_image_forwarded(self):
        client, backend = _edit_client(ImageResult(images=[_PNG]))
        resp = self._post(client)
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIsInstance(body["created"], int)
        self.assertEqual(len(body["data"]), 1)
        self.assertEqual(
            body["data"][0]["b64_json"], base64.b64encode(_PNG).decode("ascii")
        )
        # Image bytes reach the backend; absent mask becomes None.
        self.assertEqual(backend.calls[0]["image"], _png_bytes())
        self.assertIsNone(backend.calls[0]["mask"])

    def test_url_format_returns_data_url(self):
        client, _ = _edit_client(ImageResult(images=[_PNG]))
        resp = self._post(client, data={"response_format": "url"})
        entry = resp.json()["data"][0]
        self.assertEqual(
            entry["url"],
            "data:image/png;base64," + base64.b64encode(_PNG).decode("ascii"),
        )

    def test_extension_fields_forwarded_with_parsed_size(self):
        client, backend = _edit_client(ImageResult(images=[_PNG]))
        self._post(client, data={
            "n": "2", "size": "512x768", "num_inference_steps": "8",
            "guidance_scale": "0.0", "negative_prompt": "blurry",
            "strength": "0.6", "seed": "7",
        })
        call = backend.calls[0]
        self.assertEqual(call["n"], 2)
        self.assertEqual((call["width"], call["height"]), (512, 768))
        self.assertEqual(call["num_inference_steps"], 8)
        self.assertEqual(call["guidance_scale"], 0.0)
        self.assertEqual(call["negative_prompt"], "blurry")
        self.assertEqual(call["strength"], 0.6)
        self.assertEqual(call["seed"], 7)

    def test_omitted_optionals_forwarded_as_none_and_defaults(self):
        client, backend = _edit_client(ImageResult(images=[_PNG]))
        self._post(client)
        call = backend.calls[0]
        self.assertEqual(call["n"], 1)
        self.assertEqual(
            (call["width"], call["height"], call["num_inference_steps"],
             call["guidance_scale"], call["negative_prompt"],
             call["strength"], call["seed"]),
            (None, None, None, None, None, None, None),
        )

    def test_mask_upload_is_forwarded(self):
        client, backend = _edit_client(ImageResult(images=[_PNG]))
        self._post(client, files={
            "image": ("in.png", _png_bytes(), "image/png"),
            "mask": ("m.png", b"mask-bytes", "image/png"),
        })
        self.assertEqual(backend.calls[0]["mask"], b"mask-bytes")

    def test_bad_size_is_400_without_calling_backend(self):
        client, backend = _edit_client(ImageResult(images=[_PNG]))
        resp = self._post(client, data={"size": "1024"})
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["error"]["code"], "invalid_size")
        self.assertEqual(backend.calls, [])

    def test_invalid_image_returns_400(self):
        # Upload-decoding failures (InvalidImage) surface as 400 at the route.
        client, _ = _edit_client(InvalidImage("bad image"))
        resp = self._post(client)
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["error"]["code"], "invalid_image")

    def test_unsupported_model_maps_to_400(self):
        client, _ = _edit_client(UnsupportedModel("not a diffusion model"))
        resp = self._post(client)
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["error"]["code"], "unsupported_model")

    def test_model_not_found_maps_to_404(self):
        client, _ = _edit_client(ModelNotFound("org/ghost", Path("/nowhere")))
        resp = self._post(client, data={"model": "org/ghost"})
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.json()["error"]["code"], "model_not_found")

    def test_n_less_than_one_is_422(self):
        client, backend = _edit_client(ImageResult(images=[_PNG]))
        resp = self._post(client, data={"n": "0"})
        self.assertEqual(resp.status_code, 422)
        self.assertEqual(backend.calls, [])


if __name__ == "__main__":
    unittest.main()
