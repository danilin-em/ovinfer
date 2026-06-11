# Image2ImagePipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `og.Image2ImagePipeline` to the no-mask branch of `POST /v1/images/edits`, replacing the 501 stub with real inference (multipart image upload + prompt -> b64/data-URL PNG output, with an `strength` knob and the same diffusion passthroughs as Text2Image).

**Architecture:** Sync core (`Backend.image_edit`) resolves the model, routes a non-empty mask to a 501 (Inpainting is a separate pipeline), rejects non-diffusion models with 400, decodes the uploaded image into an `ov.Tensor`, builds the og pipeline (kind `"image2image"`), runs one blocking `generate(prompt, image, **kwargs)`, and returns PNG bytes (`ImageResult`). The FastAPI transport reads the multipart form, parses `size`, forwards optional diffusion fields (incl. `strength`), and wraps the bytes into the OpenAI images envelope. Dispatch is fixed by route (intent is "edit"); the mask split lives in the backend.

**Tech Stack:** openvino-genai 2026.2.0, FastAPI + python-multipart, Pillow/numpy/openvino (already deps), Python `unittest`.

**Spec:** `docs/superpowers/specs/2026-06-11-image2image-design.md`

**Conventions for this codebase:**
- Code, comments, docstrings, commits — **English only** (CLAUDE.md). Spec/plan prose is Russian; this plan's code blocks are the source of truth.
- Tests are `unittest` (not pytest). Run a file: `.venv/bin/python -m unittest tests.test_image2image -v`. Run one class: `.venv/bin/python -m unittest tests.test_image2image.ClassName -v`.
- `og` (openvino_genai) is never imported in tests: pure helpers are tested directly (`openvino`/`PIL`/`numpy` are fine — they are core deps), dispatch tests stub `Backend._pipeline`, transport tests stub the whole backend. Mirror `tests/test_text2image.py` and `tests/test_vlm.py`.
- **Reused, do not recreate:** `ImageResult`, `_encode_pngs`, `Backend.image_generate` (pattern), `server._parse_size`, `server._image_entry`, `errors.UnsupportedModel`, `registry.is_diffusion_model`, the `_ERROR_MAP` entries (all four domain errors already mapped). This plan only adds the image2image-specific pieces.
- **Branch:** do this work on `feat/image2image-pipeline` (create off `main`; the spec is already on `main`). If executing in an isolated worktree, create it via `superpowers:using-git-worktrees` first.

---

## File Structure

- `ovinfer/config.py` — **modify**: add `"image2image"` to `PIPELINE_KINDS` (line 14).
- `ovinfer/backend.py` — **modify**: refactor `_load_image_tensor` into a byte-level `_decode_image_tensor` helper; add an `"image2image"` branch to `_create_pipeline`; replace the `image_edit` 501 stub with a real implementation (moved next to `image_generate`).
- `ovinfer/server.py` — **modify**: rewrite the `images_edits` route (extension Form fields + `_parse_size` + envelope via `_image_entry`); update the module docstring map line for Image2Image.
- `tests/test_image2image.py` — **create**: all unit/dispatch/transport tests.
- `CLAUDE.md` — **modify**: add Image2Image to the "Состояние" section; bump the section date to 2026-06-12.

**No changes needed** (verified against the spec): `ovinfer/errors.py` (all four domain errors exist), `server._ERROR_MAP` (all mapped), `ovinfer/npu.py` (`reject_reason` falls through to `None` for `"image2image"`), `registry.is_diffusion_model` (marker `model_index.json` is shared with Text2Image).

---

## Task 1: `image2image` pipeline kind in config

**Files:**
- Modify: `ovinfer/config.py:14` (`PIPELINE_KINDS`)
- Test: `tests/test_image2image.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_image2image.py`:

```python
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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests.test_image2image.PipelineKindTest -v`
Expected: FAIL — `parse_preload` raises `ConfigError` (kind not in `PIPELINE_KINDS`), and the membership assertion fails.

- [ ] **Step 3: Write minimal implementation**

In `ovinfer/config.py`, change line 14:

```python
PIPELINE_KINDS = ("llm", "vlm", "embedding", "rerank", "text2image", "image2image")
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m unittest tests.test_image2image.PipelineKindTest -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add ovinfer/config.py tests/test_image2image.py
git commit -m "feat: register image2image pipeline kind for preload"
```

---

## Task 2: `_decode_image_tensor` byte-level decoder (DRY refactor)

Extract the PIL-decode core of `_load_image_tensor` into a byte-level helper so
both the VLM path (URL fetch) and the image-edit path (multipart upload) share
one decode + one `InvalidImage` site.

**Files:**
- Modify: `ovinfer/backend.py:158-173` (`_load_image_tensor`)
- Test: `tests/test_image2image.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_image2image.py` (before the `if __name__` block):

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests.test_image2image.DecodeImageTensorTest -v`
Expected: FAIL with `ImportError: cannot import name '_decode_image_tensor' from 'ovinfer.backend'`

- [ ] **Step 3: Write minimal implementation**

In `ovinfer/backend.py`, replace the whole `_load_image_tensor` function (currently lines 158-173):

```python
def _load_image_tensor(url: str) -> Any:
    """Fetch and decode an image URL into an ov.Tensor (uint8 RGB, HWC)."""
    import io

    import numpy as np
    import openvino as ov
    from PIL import Image, UnidentifiedImageError

    raw = _fetch_image_bytes(url)
    try:
        pic = Image.open(io.BytesIO(raw))
        pic.load()  # force decode now: truncated/broken data raises OSError here
        pic = pic.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImage(f"could not decode image: {exc}") from exc
    return ov.Tensor(np.array(pic))
```

with:

```python
def _decode_image_tensor(raw: bytes) -> Any:
    """Decode raw image bytes into an ov.Tensor (uint8 RGB, HWC).

    pic.load() forces a full decode now so truncated/broken data raises here,
    not lazily downstream. Shared by the VLM path (via _load_image_tensor, after
    a URL fetch) and the image-edit path (raw multipart upload bytes).
    """
    import io

    import numpy as np
    import openvino as ov
    from PIL import Image, UnidentifiedImageError

    try:
        pic = Image.open(io.BytesIO(raw))
        pic.load()  # force decode now: truncated/broken data raises OSError here
        pic = pic.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImage(f"could not decode image: {exc}") from exc
    return ov.Tensor(np.array(pic))


def _load_image_tensor(url: str) -> Any:
    """Fetch and decode an image URL into an ov.Tensor (uint8 RGB, HWC)."""
    return _decode_image_tensor(_fetch_image_bytes(url))
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m unittest tests.test_image2image.DecodeImageTensorTest -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Run the VLM suite to confirm no regression**

Run: `.venv/bin/python -m unittest tests.test_vlm -v`
Expected: PASS (the existing `LoadImageTensorTest` still passes via the refactored entry point).

- [ ] **Step 6: Commit**

```bash
git add ovinfer/backend.py tests/test_image2image.py
git commit -m "refactor: extract _decode_image_tensor from _load_image_tensor"
```

---

## Task 3: `Backend.image_edit` dispatch + kwargs + mask routing

**Files:**
- Modify: `ovinfer/backend.py` — add `"image2image"` branch in `_create_pipeline` (after the `text2image` branch, ~line 324); add the real `image_edit` after `image_generate` (~line 606) under a new section header; delete the old `image_edit` 501 stub (lines 613-629).
- Test: `tests/test_image2image.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_image2image.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests.test_image2image.ImageEditDispatchTest -v`
Expected: FAIL — the current `image_edit` stub raises `PipelineNotImplemented` for the no-mask case and has a different signature (`n, size, response_format`), so the keyword calls error / the no-mask path never returns an `ImageResult`.

- [ ] **Step 3: Write minimal implementation**

In `ovinfer/backend.py`, add a branch in `_create_pipeline` right after the `text2image` branch (after `return og.Text2ImagePipeline(model_dir, device)`, before `raise ValueError`):

```python
        if kind == "image2image":
            return og.Image2ImagePipeline(model_dir, device)
```

Add the real `image_edit` immediately after `image_generate` returns (after the line `return ImageResult(images=_encode_pngs(tensor))` that ends `image_generate`, ~line 606), introducing its own section header:

```python
    # -- image editing ------------------------------------------------------------

    def image_edit(
        self,
        model_id: str,
        prompt: str,
        image: bytes,
        mask: bytes | None,
        *,
        n: int = 1,
        width: int | None = None,
        height: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        negative_prompt: str | None = None,
        strength: float | None = None,
        seed: int | None = None,
    ) -> ImageResult:
        """Image-to-image editing via og.Image2ImagePipeline (no-mask branch).

        Check order tracks the cost of each step: model resolution (404) first;
        a non-empty mask is Inpainting (501, a separate pipeline) before any
        diffusion work; the diffusion-model check (400) before the cheap input
        decode (400); the input image is decoded before the expensive pipeline
        build, so a broken upload 400s without loading the model. Only
        explicitly-set diffusion params are forwarded, so og's own defaults
        (including strength=0.8) are not clobbered; guidance_scale/strength=0.0
        pass through (meaningful values for CFG-distilled / preserve-input runs).
        """
        model_dir = registry.resolve_model_dir(self.settings.models_root, model_id)
        if not mask:
            # Zero-length upload counts as "no mask" to avoid silently picking
            # the inpainting path with an empty mask tensor.
            mask = None
        if mask is not None:
            raise PipelineNotImplemented("InpaintingPipeline")
        if not registry.is_diffusion_model(model_dir):
            raise UnsupportedModel(
                f"model '{model_id}' is not a diffusion model and cannot edit images"
            )
        image_tensor = _decode_image_tensor(image)
        kwargs: dict[str, Any] = {"num_images_per_prompt": n}
        if width is not None:
            kwargs["width"] = width
        if height is not None:
            kwargs["height"] = height
        if num_inference_steps is not None:
            kwargs["num_inference_steps"] = num_inference_steps
        if guidance_scale is not None:
            kwargs["guidance_scale"] = guidance_scale
        if negative_prompt is not None:
            kwargs["negative_prompt"] = negative_prompt
        if strength is not None:
            kwargs["strength"] = strength
        if seed is not None:
            kwargs["rng_seed"] = seed
        with self._pipeline(model_id, "image2image") as pipe:
            tensor = pipe.generate(prompt, image_tensor, **kwargs)
        return ImageResult(images=_encode_pngs(tensor))
```

Delete the old `image_edit` stub from the "not implemented in v0.1 (501 contract)" section (currently lines 613-629):

```python
    def image_edit(
        self,
        model_id: str,
        prompt: str,
        image: bytes,
        mask: bytes | None,
        n: int = 1,
        size: str | None = None,
        response_format: str = "b64_json",
    ) -> dict:
        self._resolve(model_id)
        if not mask:
            # Zero-length upload counts as "no mask" to avoid silently picking
            # the inpainting path with an empty mask tensor.
            mask = None
        pipeline = "InpaintingPipeline" if mask is not None else "Image2ImagePipeline"
        raise PipelineNotImplemented(pipeline)
```

(Leave `_resolve` and the remaining stubs — `transcribe`, `translate`, `speech`, `video_generate` — untouched; they still use `_resolve`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m unittest tests.test_image2image.ImageEditDispatchTest -v`
Expected: PASS (8 tests)

- [ ] **Step 5: Commit**

```bash
git add ovinfer/backend.py tests/test_image2image.py
git commit -m "feat: implement Backend.image_edit via Image2ImagePipeline"
```

---

## Task 4: Transport route + multipart contract

**Files:**
- Modify: `ovinfer/server.py` — rewrite the `images_edits` route (lines 391-405); update the module docstring map line for Image2Image (line 15).
- Test: `tests/test_image2image.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_image2image.py`:

```python
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

    def test_non_empty_mask_501_maps_to_501(self):
        client, _ = _edit_client(PipelineNotImplemented("InpaintingPipeline"))
        resp = self._post(client, files={
            "image": ("in.png", _png_bytes(), "image/png"),
            "mask": ("m.png", b"mask-bytes", "image/png"),
        })
        self.assertEqual(resp.status_code, 501)
        self.assertEqual(resp.json()["error"]["code"], "pipeline_not_implemented")

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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests.test_image2image.ImagesEditsRouteTest -v`
Expected: FAIL — the current route passes positional `(n, size, response_format)` to `image_edit` and returns its raw value; size is not parsed, the envelope is not built, `n` has no `ge=1`, and the extension fields don't exist.

- [ ] **Step 3: Write minimal implementation**

In `ovinfer/server.py`, replace the `images_edits` route (lines 391-405):

```python
    @app.post("/v1/images/edits")
    def images_edits(
        model: str = Form(...),
        prompt: str = Form(...),
        image: UploadFile = File(...),
        mask: UploadFile | None = File(None),
        n: int = Form(1),
        size: str | None = Form(None),
        response_format: str = Form("b64_json"),
    ) -> Any:
        image_bytes = image.file.read()
        mask_bytes = mask.file.read() if mask is not None else None
        return backend.image_edit(
            model, prompt, image_bytes, mask_bytes, n, size, response_format
        )
```

with:

```python
    @app.post("/v1/images/edits")
    def images_edits(
        model: str = Form(...),
        prompt: str = Form(...),
        image: UploadFile = File(...),
        mask: UploadFile | None = File(None),
        n: int = Form(1, ge=1),
        size: str | None = Form(None),
        response_format: str = Form("b64_json"),
        # Diffusion knobs (not in OpenAI's edits schema): on multipart there is
        # no extra="allow", so each is declared explicitly and forwarded when set.
        num_inference_steps: int | None = Form(None),
        guidance_scale: float | None = Form(None),
        negative_prompt: str | None = Form(None),
        strength: float | None = Form(None),
        seed: int | None = Form(None),
    ) -> Any:
        try:
            width, height = _parse_size(size)
        except ValueError as exc:
            return JSONResponse(
                status_code=400,
                content=_error_body(
                    str(exc), "invalid_request_error", "invalid_size"
                ),
            )
        image_bytes = image.file.read()
        mask_bytes = mask.file.read() if mask is not None else None
        result = backend.image_edit(
            model,
            prompt,
            image_bytes,
            mask_bytes,
            n=n,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            strength=strength,
            seed=seed,
        )
        return {
            "created": int(time.time()),
            "data": [
                _image_entry(png, response_format) for png in result.images
            ],
        }
```

Update the module docstring map line near the top of `server.py` (line 15):

```
    Image2ImagePipeline        -> POST /v1/images/edits              (implemented)
```

(Leave the next line, `InpaintingPipeline ... (mask) (501)`, as is.)

- [ ] **Step 4: Run the full test file to verify everything passes**

Run: `.venv/bin/python -m unittest tests.test_image2image -v`
Expected: PASS (PipelineKind, DecodeImageTensor, ImageEditDispatch, ImagesEditsRoute).

- [ ] **Step 5: Run the whole suite to check for regressions**

Run: `.venv/bin/python -m unittest discover -s tests -v`
Expected: PASS (existing test_embeddings, test_npu, test_rerank, test_vlm, test_text2image, plus test_image2image).

- [ ] **Step 6: Commit**

```bash
git add ovinfer/server.py tests/test_image2image.py
git commit -m "feat: wire POST /v1/images/edits (no-mask) to image_edit"
```

---

## Task 5: Update project state docs

**Files:**
- Modify: `CLAUDE.md` ("## Состояние" section)

- [ ] **Step 1: Add the Image2Image state entry**

In `CLAUDE.md`, under "## Состояние", add after the Text2Image bullet (Russian prose, consistent with the existing entries):

```markdown
- **Image2Image реализован 2026-06-12**: `og.Image2ImagePipeline` на
  `POST /v1/images/edits` (ветка без mask) — multipart-загрузка `image` +
  `prompt` отдаёт PNG (`b64_json`/data-URL); диффузионные «ручки»
  (`num_inference_steps`, `guidance_scale`, `negative_prompt`, `seed`) плюс
  img2img-knob `strength` как опциональные Form-поля-расширения. Декод входа —
  общий с VLM хелпер `_decode_image_tensor`. Непустой `mask` → 501
  `InpaintingPipeline` (отдельная итерация); не-диффузионная модель → 400.
  **Требует `vae_encoder`** (пиксели входа → латент), в отличие от Text2Image.
  Приёмка на `OpenVINO/LCM_Dreamshaper_v7-int8-ov` (есть `vae_encoder`). Спека:
  `docs/superpowers/specs/2026-06-11-image2image-design.md`; план:
  `docs/superpowers/plans/2026-06-12-image2image.md`.
```

- [ ] **Step 2: Bump the section date marker**

Change the heading `## Состояние (2026-06-11, обновлять при смене фазы)` to:

```markdown
## Состояние (2026-06-12, обновлять при смене фазы)
```

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: record Image2Image implementation in project state"
```

---

## Task 6: Acceptance on the real model

**Manual verification — no code, no commit. Run via `openvino-env` (GPU/NPU libs) on a spare port (the user keeps a server on :8000). The acceptance curls need loopback access; if the Bash sandbox blocks it, disable the sandbox for those commands (see memory `acceptance-testing-http-server`).**

- [ ] **Step 1: Create an input image**

```bash
.venv/bin/python -c "from PIL import Image; Image.new('RGB',(512,512),(80,120,200)).save('/tmp/ovinfer_input.png')"
file /tmp/ovinfer_input.png
```

Expected: `file` reports `PNG image data, 512 x 512`.

- [ ] **Step 2: Start the server on a spare port**

Run (background or a second terminal):

```bash
cd /home/evgenii/Projects/ovinfer
OVINFER_MODELS_ROOT=~/.openvino/models/huggingface.co \
  openvino-env .venv/bin/ovinfer server --listen 127.0.0.1:8012 --device CPU
```

Expected: uvicorn starts; `GET http://127.0.0.1:8012/v1/models` lists `OpenVINO/LCM_Dreamshaper_v7-int8-ov`.

- [ ] **Step 3: Edit the image (img2img with strength)**

```bash
curl -s http://127.0.0.1:8012/v1/images/edits \
  -F model=OpenVINO/LCM_Dreamshaper_v7-int8-ov \
  -F prompt='a cat wearing a wizard hat, vivid colors' \
  -F image=@/tmp/ovinfer_input.png \
  -F num_inference_steps=4 \
  -F strength=0.6 \
  | python -c 'import sys,json,base64; d=json.load(sys.stdin); b=base64.b64decode(d["data"][0]["b64_json"]); open("/tmp/ovinfer_i2i.png","wb").write(b); print("created=",d["created"],"bytes=",len(b))'
file /tmp/ovinfer_i2i.png
```

Expected: `file` reports `PNG image data` (512 x 512 unless `size` overrides). Non-zero byte count.

- [ ] **Step 4: Verify response_format=url returns a data: URL**

```bash
curl -s http://127.0.0.1:8012/v1/images/edits \
  -F model=OpenVINO/LCM_Dreamshaper_v7-int8-ov \
  -F prompt='a teacup' \
  -F image=@/tmp/ovinfer_input.png \
  -F num_inference_steps=4 \
  -F response_format=url \
  | python -c 'import sys,json; print(json.load(sys.stdin)["data"][0]["url"][:40])'
```

Expected: prints `data:image/png;base64,` followed by base64.

- [ ] **Step 5: Verify a non-empty mask returns 501 (Inpainting)**

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8012/v1/images/edits \
  -F model=OpenVINO/LCM_Dreamshaper_v7-int8-ov \
  -F prompt='x' \
  -F image=@/tmp/ovinfer_input.png \
  -F mask=@/tmp/ovinfer_input.png
```

Expected: `501`.

- [ ] **Step 6: Verify a non-diffusion model is rejected with 400**

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8012/v1/images/edits \
  -F model=OpenVINO/Qwen3-0.6B-int4-ov \
  -F prompt='x' \
  -F image=@/tmp/ovinfer_input.png
```

Expected: `400`. (Substitute any local plain-LLM model id from `/v1/models` if that one is absent.)

- [ ] **Step 7: Stop the server**

Stop the uvicorn process (Ctrl-C or kill the background job).

- [ ] **Step 8: Finalize the branch**

Use the `superpowers:finishing-a-development-branch` skill to decide merge/PR/cleanup for `feat/image2image-pipeline`.

---

## Self-Review (completed during planning)

- **Spec coverage** (against `2026-06-11-image2image-design.md`):
  - og facts / `generate(prompt, image, **kwargs)` + `strength` — Task 3.
  - Route fixes intent; mask split in backend (non-empty → 501, empty → no mask) — Task 3 (dispatch) + Task 4 (transport mapping).
  - Param mapping table, only-set kwargs, `0.0` passthrough for `guidance_scale`/`strength` — Task 3.
  - `_decode_image_tensor` extracted from `_load_image_tensor`, fed raw upload bytes — Task 2 + Task 3.
  - `_create_pipeline` `image2image` branch — Task 3.
  - `image_edit` signature + check order (404 → 501 mask → 400 unsupported → 400 invalid image → build) — Task 3.
  - Transport route (Form ext-fields, `_parse_size` 400 invalid_size, `_image_entry` envelope, `n` `ge=1` → 422) — Task 4.
  - Docstring map line `(implemented)` — Task 4.
  - Errors: all four already in `_ERROR_MAP`, no `errors.py` change — confirmed in File Structure (verified, no task needed).
  - Config `PIPELINE_KINDS` + preload — Task 1.
  - NPU: `reject_reason` falls through to `None` for `"image2image"`, no `npu.py` change — confirmed in File Structure (no task needed).
  - Edge cases (empty mask, omitted size, n>1, 0.0 knobs, seed default vs set, url vs b64) — covered across Task 3/4 tests.
  - Acceptance on `LCM_Dreamshaper_v7-int8-ov` (has `vae_encoder`), spare port via `openvino-env`, mask→501 check — Task 6.
  - State docs — Task 5.
  No gaps.
- **Placeholders:** none — every code step shows complete code and exact commands.
- **Type consistency:** `image_edit(model_id, prompt, image: bytes, mask: bytes | None, *, n, width, height, num_inference_steps, guidance_scale, negative_prompt, strength, seed) -> ImageResult`; `_decode_image_tensor(raw: bytes) -> ov.Tensor`; `_load_image_tensor(url) -> ov.Tensor` (now delegates); `ImageResult.images: list[bytes]`; `_encode_pngs(tensor) -> list[bytes]`; `_parse_size(str|None) -> (int|None, int|None)`; `_image_entry(bytes, str) -> dict`. The transport stub backend's `image_edit` signature (image/mask positional, knobs keyword-only) matches the real one and the route's call. Consistent across tasks.
```
