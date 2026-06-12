# InpaintingPipeline Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Wire `og.InpaintingPipeline` to the mask branch of `POST /v1/images/edits`, replacing the 501 stub with real inference (multipart `image` + `mask` + `prompt` -> b64/data-URL PNG output, with hybrid OpenAI/og mask semantics and the same diffusion knobs as Image2Image).

**Architecture:** Sync core (`Backend.image_edit`) resolves the model, rejects non-diffusion models with 400 (now covering both edit branches), decodes the uploaded image into a batched `ov.Tensor`, then branches on the mask: a non-empty mask is decoded into an og mask tensor and routed to `og.InpaintingPipeline` (`generate(prompt, image, mask_image, **kwargs)`, kind `"inpainting"`); an empty/absent mask stays on `og.Image2ImagePipeline`. The FastAPI transport is unchanged (it already forwards the mask bytes); only its docstring map line flips to `(implemented)`.

**Tech Stack:** openvino-genai 2026.2.0, FastAPI + python-multipart, Pillow/numpy/openvino (already deps), Python `unittest`.

**Spec:** `docs/superpowers/specs/2026-06-12-inpainting-design.md`

**Conventions for this codebase:**
- Code, comments, docstrings, commits — **English only** (CLAUDE.md). Spec/plan prose is Russian; this plan's code blocks are the source of truth.
- Tests are `unittest` (not pytest). Run a file: `.venv/bin/python -m unittest tests.test_inpainting -v`. Run one class: `.venv/bin/python -m unittest tests.test_inpainting.ClassName -v`.
- `og` (openvino_genai) is never imported in tests: pure helpers are tested directly (`openvino`/`PIL`/`numpy` are fine — core deps), dispatch tests stub `Backend._pipeline`, transport tests stub the whole backend. Mirror `tests/test_image2image.py`.
- **Reused, do not recreate:** `ImageResult`, `_encode_pngs`, `_decode_image_tensor(raw, batched=True)`, `Backend.image_edit` (extended in place), `server._image_entry`, `server._parse_size`, `errors.UnsupportedModel`/`InvalidImage`, `registry.is_diffusion_model`, the `_ERROR_MAP` entries (all four domain errors already mapped). This plan adds only the inpainting-specific pieces.
- **Branch:** do this work on `feat/inpainting-pipeline` (create off `main`; the spec is already on `main`). If executing in an isolated worktree, create it via `superpowers:using-git-worktrees` first.

---

## File Structure

- `ovinfer/config.py` — **modify**: add `"inpainting"` to `PIPELINE_KINDS` (line 14).
- `ovinfer/backend.py` — **modify**: add the `_decode_mask_tensor` helper (after `_load_image_tensor`); add an `"inpainting"` branch to `_create_pipeline`; rewrite `image_edit` to move the diffusion check above the mask split and add the inpainting branch.
- `ovinfer/server.py` — **modify**: flip the module docstring map line for Inpainting to `(implemented)` (line 16). The `images_edits` route needs **no code change** (it already reads and forwards the mask bytes).
- `tests/test_inpainting.py` — **create**: all inpainting unit/dispatch/transport tests.
- `tests/test_image2image.py` — **modify**: update the three tests pinned to the old "mask -> 501" ordering (see Task 3 / Task 4).
- `CLAUDE.md` — **modify**: add Inpainting to the "Состояние" section; keep the section date at 2026-06-12.

**No changes needed** (verified against the spec): `ovinfer/errors.py` (all four domain errors exist), `server._ERROR_MAP` (all mapped), `ovinfer/npu.py` (`reject_reason` falls through to `None` for `"inpainting"`), `registry.is_diffusion_model` (marker `model_index.json` is shared with Image2Image).

---

## Task 1: `inpainting` pipeline kind in config

**Files:**
- Modify: `ovinfer/config.py:14` (`PIPELINE_KINDS`)
- Test: `tests/test_inpainting.py` (create)

- [ ] **Step 1: Write the failing test**

Create `tests/test_inpainting.py`:

```python
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


if __name__ == "__main__":
    unittest.main()
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests.test_inpainting.PipelineKindTest -v`
Expected: FAIL — `parse_preload` raises `ConfigError` (kind not in `PIPELINE_KINDS`), and the membership assertion fails.

- [ ] **Step 3: Write minimal implementation**

In `ovinfer/config.py`, change line 14:

```python
PIPELINE_KINDS = (
    "llm",
    "vlm",
    "embedding",
    "rerank",
    "text2image",
    "image2image",
    "inpainting",
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m unittest tests.test_inpainting.PipelineKindTest -v`
Expected: PASS (2 tests)

- [ ] **Step 5: Commit**

```bash
git add ovinfer/config.py tests/test_inpainting.py
git commit -m "feat: register inpainting pipeline kind for preload"
```

---

## Task 2: `_decode_mask_tensor` hybrid mask decoder

Decode an uploaded inpainting mask into an og `mask_image` tensor with hybrid
semantics: an RGBA mask (OpenAI style) marks the edit region with *transparent*
pixels (alpha inverted to white), a mask without alpha is read as plain
luminance (white -> repaint). Output is 4D NHWC `(1, H, W, 3)` uint8 to match
the batched image tensor.

**Files:**
- Modify: `ovinfer/backend.py` — add `_decode_mask_tensor` right after `_load_image_tensor` (currently ends at line 190), before `_encode_pngs`.
- Test: `tests/test_inpainting.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_inpainting.py` (before the `if __name__` block):

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests.test_inpainting.DecodeMaskTensorTest -v`
Expected: FAIL with `ImportError: cannot import name '_decode_mask_tensor' from 'ovinfer.backend'`

- [ ] **Step 3: Write minimal implementation**

In `ovinfer/backend.py`, add `_decode_mask_tensor` immediately after the
`_load_image_tensor` function (after its `return` on line 190), before
`_encode_pngs`:

```python
def _decode_mask_tensor(raw: bytes) -> Any:
    """Decode an inpainting mask into an og mask_image tensor: 4D NHWC (1,H,W,3)
    uint8 where white (255) marks the region to repaint, black (0) the region to
    keep (og/diffusers convention).

    Hybrid input handling: an OpenAI-style RGBA mask marks the edit region with
    *transparent* pixels, so when an alpha channel is present it is used and
    inverted (alpha 0 -> 255 white -> repaint). A mask without alpha is read as
    plain luminance (white -> repaint), matching diffusers tutorials and simple
    b/w masks. The single-channel edit map is broadcast to 3 channels so the
    mask tensor matches the image tensor layout og's preprocessing expects.
    """
    import io

    import numpy as np
    import openvino as ov
    from PIL import Image, UnidentifiedImageError

    try:
        pic = Image.open(io.BytesIO(raw))
        pic.load()  # force decode now: truncated/broken data raises here
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImage(f"could not decode mask: {exc}") from exc
    if pic.mode in ("RGBA", "LA") or "transparency" in pic.info:
        alpha = np.array(pic.convert("RGBA"))[..., 3]
        edit = 255 - alpha          # transparent (alpha 0) -> white -> repaint
    else:
        edit = np.array(pic.convert("L"))  # white -> repaint
    rgb = np.repeat(edit[..., np.newaxis], 3, axis=2).astype(np.uint8)
    return ov.Tensor(rgb[np.newaxis, ...])
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m unittest tests.test_inpainting.DecodeMaskTensorTest -v`
Expected: PASS (3 tests)

- [ ] **Step 5: Commit**

```bash
git add ovinfer/backend.py tests/test_inpainting.py
git commit -m "feat: add hybrid OpenAI/og inpainting mask decoder"
```

---

## Task 3: `image_edit` inpainting dispatch + `_create_pipeline` branch

Extend `Backend.image_edit` to route a non-empty mask to `og.InpaintingPipeline`,
moving the diffusion-model check above the mask split (Inpainting needs a
diffusion model too). This changes behavior pinned by two tests in
`tests/test_image2image.py`, updated here in the same commit.

**Files:**
- Modify: `ovinfer/backend.py` — add an `"inpainting"` branch in `_create_pipeline` (after the `image2image` branch, line 343); rewrite `image_edit` (lines 629-685).
- Modify: `tests/test_image2image.py` — delete the obsolete `test_non_empty_mask_is_501_inpainting_without_building_pipeline`; change `test_mask_beats_non_diffusion_model` to assert the new 400; fix one stale comment.
- Test: `tests/test_inpainting.py`

- [ ] **Step 1: Write the failing test**

Append to `tests/test_inpainting.py`:

```python
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
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests.test_inpainting.InpaintEditDispatchTest -v`
Expected: FAIL — the current `image_edit` raises `PipelineNotImplemented("InpaintingPipeline")` for any non-empty mask, so the inpainting dispatch never happens and the kind/shape/kwargs assertions error.

- [ ] **Step 3: Write minimal implementation**

In `ovinfer/backend.py`, add a branch in `_create_pipeline` right after the
`image2image` branch (after `return og.Image2ImagePipeline(model_dir, device)`,
line 343, before `raise ValueError`):

```python
        if kind == "inpainting":
            return og.InpaintingPipeline(model_dir, device)
```

Then replace the entire `image_edit` method (lines 629-685) with:

```python
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
        """Image editing: Image2Image (no mask) or Inpainting (mask) via og.

        Check order tracks the cost of each step: model resolution (404) first;
        the diffusion-model check (400) covers both branches (Inpainting needs a
        diffusion model too); the input image — and, for inpainting, the mask —
        are decoded (cheap, 400 on bad bytes) before the expensive pipeline
        build, so a broken upload 400s without loading the model. A non-empty
        mask routes to og.InpaintingPipeline (mask_image is the third positional
        arg); an empty (zero-length) mask upload counts as "no mask" ->
        Image2Image. Only explicitly-set diffusion params are forwarded, so og's
        own defaults (including strength=0.8) are not clobbered;
        guidance_scale/strength=0.0 pass through (meaningful values for
        CFG-distilled / preserve-input runs).
        """
        model_dir = registry.resolve_model_dir(self.settings.models_root, model_id)
        if not mask:
            # Zero-length upload counts as "no mask" to avoid silently picking
            # the inpainting path with an empty mask tensor.
            mask = None
        if not registry.is_diffusion_model(model_dir):
            raise UnsupportedModel(
                f"model '{model_id}' is not a diffusion model and cannot edit images"
            )
        image_tensor = _decode_image_tensor(image, batched=True)
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
        if mask is not None:
            mask_tensor = _decode_mask_tensor(mask)
            with self._pipeline(model_id, "inpainting") as pipe:
                tensor = pipe.generate(prompt, image_tensor, mask_tensor, **kwargs)
        else:
            with self._pipeline(model_id, "image2image") as pipe:
                tensor = pipe.generate(prompt, image_tensor, **kwargs)
        return ImageResult(images=_encode_pngs(tensor))
```

(`PipelineNotImplemented` is still imported and used by the remaining 501 stubs —
`transcribe`/`translate`/`speech`/`video_generate` — so its import stays.)

- [ ] **Step 4: Run the inpainting dispatch test to verify it passes**

Run: `.venv/bin/python -m unittest tests.test_inpainting.InpaintEditDispatchTest -v`
Expected: PASS (7 tests)

- [ ] **Step 5: Update the two pinned tests in `tests/test_image2image.py`**

The diffusion check now precedes the mask split, so two `ImageEditDispatchTest`
tests assert obsolete behavior. Delete `test_non_empty_mask_is_501_inpainting_without_building_pipeline` entirely:

```python
    def test_non_empty_mask_is_501_inpainting_without_building_pipeline(self):
        root, model = _diffusion_root()
        be = _DispatchBackend(root, _FakeI2IPipe())
        with self.assertRaises(PipelineNotImplemented) as ctx:
            be.image_edit(model, "x", _png_bytes(), b"\x89PNG-mask")
        self.assertEqual(ctx.exception.pipeline, "InpaintingPipeline")
        self.assertEqual(be.pipeline_requests, [])
```

Replace `test_mask_beats_non_diffusion_model` with a test of the new ordering:

```python
    def test_mask_beats_non_diffusion_model(self):
        # A non-empty mask is always Inpainting (501): the mask check precedes
        # the diffusion check, so a mask on a non-diffusion model is 501, not 400.
        root, model = _llm_root()
        be = _DispatchBackend(root, _FakeI2IPipe())
        with self.assertRaises(PipelineNotImplemented):
            be.image_edit(model, "x", _png_bytes(), b"mask")
        self.assertEqual(be.pipeline_requests, [])
```

becomes:

```python
    def test_non_diffusion_model_is_unsupported_even_with_mask(self):
        # The diffusion check precedes the mask split: a mask on a non-diffusion
        # model is 400 UnsupportedModel (Inpainting needs a diffusion model too),
        # not 501.
        root, model = _llm_root()
        be = _DispatchBackend(root, _FakeI2IPipe())
        with self.assertRaises(UnsupportedModel):
            be.image_edit(model, "x", _png_bytes(), b"mask")
        self.assertEqual(be.pipeline_requests, [])
```

Then fix the stale comment in `test_unknown_model_404s_before_anything_else`:

```python
        # Non-empty mask too: model resolution must win the 404 over the mask 501.
```

becomes:

```python
        # Non-empty mask too: model resolution must win the 404 over the
        # mask/diffusion checks.
```

If `PipelineNotImplemented` is now unused in `tests/test_image2image.py`, remove
it from the import line `from ovinfer.errors import ...` to keep the import
clean. (Verify with `grep -n PipelineNotImplemented tests/test_image2image.py`
after the edits — if the transport test in Task 4 still references it, leave it.)

- [ ] **Step 6: Run both image-edit test files to verify green**

Run: `.venv/bin/python -m unittest tests.test_inpainting tests.test_image2image -v`
Expected: PASS (inpainting: PipelineKind + DecodeMaskTensor + InpaintEditDispatch; image2image: all, with the two tests updated).

- [ ] **Step 7: Commit**

```bash
git add ovinfer/backend.py tests/test_inpainting.py tests/test_image2image.py
git commit -m "feat: route masked image edits to InpaintingPipeline"
```

---

## Task 4: Transport contract (mask -> 200) + docstring

The `images_edits` route already reads `mask.file.read()` and forwards the bytes,
so no route code changes. This task locks the end-to-end contract (a mask upload
now yields a 200 image envelope), removes the stale transport 501 test, and flips
the docstring map line.

**Files:**
- Modify: `ovinfer/server.py:16` (module docstring map line for Inpainting).
- Modify: `tests/test_image2image.py` — remove the stale `test_non_empty_mask_returns_501`.
- Test: `tests/test_inpainting.py`

- [ ] **Step 1: Write the contract test**

Append to `tests/test_inpainting.py`:

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


class InpaintRouteTest(unittest.TestCase):
    def test_mask_upload_yields_200_envelope_and_forwards_mask_bytes(self):
        client, backend = _edit_client(ImageResult(images=[_PNG]))
        resp = client.post(
            "/v1/images/edits",
            data={"model": MODEL, "prompt": "a cat", "strength": "0.6"},
            files={
                "image": ("in.png", _png_bytes(), "image/png"),
                "mask": ("m.png", b"mask-bytes", "image/png"),
            },
        )
        self.assertEqual(resp.status_code, 200)
        body = resp.json()
        self.assertIsInstance(body["created"], int)
        self.assertEqual(
            body["data"][0]["b64_json"], base64.b64encode(_PNG).decode("ascii")
        )
        # Mask bytes reach the backend; the strength knob is forwarded.
        self.assertEqual(backend.calls[0]["mask"], b"mask-bytes")
        self.assertEqual(backend.calls[0]["strength"], 0.6)

    def test_unsupported_model_with_mask_maps_to_400(self):
        client, _ = _edit_client(UnsupportedModel("not a diffusion model"))
        resp = client.post(
            "/v1/images/edits",
            data={"model": MODEL, "prompt": "x"},
            files={
                "image": ("in.png", _png_bytes(), "image/png"),
                "mask": ("m.png", b"mask-bytes", "image/png"),
            },
        )
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["error"]["code"], "unsupported_model")
```

- [ ] **Step 2: Run the contract test**

Run: `.venv/bin/python -m unittest tests.test_inpainting.InpaintRouteTest -v`
Expected: PASS (2 tests) — the route already forwards the mask and builds the
envelope; these lock that contract now that the backend supports masks
end-to-end.

- [ ] **Step 3: Remove the stale transport 501 test from `tests/test_image2image.py`**

The masked-edit-returns-501 scenario no longer reflects reality (a mask now
yields a 200 envelope). Delete `test_non_empty_mask_returns_501` from
`ImagesEditsRouteTest`:

```python
    def test_non_empty_mask_returns_501(self):
        client, _ = _edit_client(PipelineNotImplemented("InpaintingPipeline"))
        resp = self._post(client, files={
            "image": ("in.png", _png_bytes(), "image/png"),
            "mask": ("m.png", b"mask-bytes", "image/png"),
        })
        self.assertEqual(resp.status_code, 501)
        self.assertEqual(resp.json()["error"]["code"], "pipeline_not_implemented")
```

After deleting, re-check the `PipelineNotImplemented` import in
`tests/test_image2image.py`: `grep -n PipelineNotImplemented tests/test_image2image.py`.
If there are no remaining references, drop it from the
`from ovinfer.errors import (...)` line.

- [ ] **Step 4: Flip the docstring map line in `ovinfer/server.py`**

Change line 16:

```
    InpaintingPipeline         -> POST /v1/images/edits (mask)       (501)
```

to:

```
    InpaintingPipeline         -> POST /v1/images/edits (mask)       (implemented)
```

- [ ] **Step 5: Run the whole suite to check for regressions**

Run: `.venv/bin/python -m unittest discover -s tests -v`
Expected: PASS (test_embeddings, test_npu, test_rerank, test_vlm, test_text2image, test_image2image, test_inpainting).

- [ ] **Step 6: Commit**

```bash
git add ovinfer/server.py tests/test_inpainting.py tests/test_image2image.py
git commit -m "feat: lock POST /v1/images/edits mask contract; mark Inpainting implemented"
```

---

## Task 5: Update project state docs

**Files:**
- Modify: `CLAUDE.md` ("## Состояние" section)

- [ ] **Step 1: Add the Inpainting state entry**

In `CLAUDE.md`, under "## Состояние", add after the Image2Image bullet (Russian
prose, consistent with the existing entries):

```markdown
- **Inpainting реализован 2026-06-12**: `og.InpaintingPipeline` на
  `POST /v1/images/edits` (ветка с непустым `mask`) — multipart-загрузка `image`
  + `mask` + `prompt` отдаёт PNG (`b64_json`/data-URL); те же диффузионные
  «ручки», что у Image2Image (включая `strength`). `generate(prompt, image,
  mask_image, **kwargs)` — маска третьим позиционным тензором. Декод маски —
  новый хелпер `_decode_mask_tensor` с **гибридной семантикой**: есть alpha
  (OpenAI-SDK) → прозрачное = править (инверсия alpha); нет alpha → яркость как
  og-маска (белое = править); форма `(1,H,W,3)`. Проверка `is_diffusion_model`
  переехала выше развилки `mask`, поэтому не-диффузионная модель → 400 для обеих
  веток edit (это отменяет прежний порядок «mask→501 над non-diffusion→400»).
  Требует `vae_encoder`, как Image2Image. Спека:
  `docs/superpowers/specs/2026-06-12-inpainting-design.md`; план:
  `docs/superpowers/plans/2026-06-12-inpainting.md`.
```

- [ ] **Step 2: Update the Image2Image bullet's mask note**

In the existing Image2Image bullet, the clause "Непустой `mask` → 501
`InpaintingPipeline` (отдельная итерация)" is now stale. Change it to:

```markdown
Непустой `mask` → Inpainting (реализован, см. ниже);
```

(Keep the rest of the Image2Image bullet unchanged.)

- [ ] **Step 3: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: record Inpainting implementation in project state"
```

---

## Task 6: Acceptance on the real model

**Manual verification — no code, no commit. Run via `openvino-env` (GPU/NPU libs) on a spare port (the user keeps a server on :8000). The acceptance curls need loopback access; if the Bash sandbox blocks it, disable the sandbox for those commands (see memory `acceptance-testing-http-server`).**

- [ ] **Step 1: Create an input image and an RGBA mask**

```bash
.venv/bin/python - <<'PY'
from PIL import Image, ImageDraw
# 512x512 input.
Image.new('RGB', (512, 512), (80, 120, 200)).save('/tmp/ovinfer_input.png')
# RGBA mask: opaque (keep) everywhere, transparent (edit) in a center circle.
mask = Image.new('RGBA', (512, 512), (0, 0, 0, 255))
d = ImageDraw.Draw(mask)
d.ellipse([160, 160, 352, 352], fill=(0, 0, 0, 0))  # alpha 0 -> edit region
mask.save('/tmp/ovinfer_mask.png')
print('input + mask written')
PY
file /tmp/ovinfer_input.png /tmp/ovinfer_mask.png
```

Expected: `file` reports both as `PNG image data, 512 x 512` (mask `RGBA`).

- [ ] **Step 2: Start the server on a spare port**

Run (background or a second terminal):

```bash
cd /home/evgenii/Projects/ovinfer
OVINFER_MODELS_ROOT=~/.openvino/models/huggingface.co \
  openvino-env .venv/bin/ovinfer server --listen 127.0.0.1:8013 --device CPU
```

Expected: uvicorn starts; `GET http://127.0.0.1:8013/v1/models` lists `OpenVINO/LCM_Dreamshaper_v7-int8-ov`.

- [ ] **Step 3: Inpaint with an RGBA mask (transparent region edited)**

```bash
curl -s http://127.0.0.1:8013/v1/images/edits \
  -F model=OpenVINO/LCM_Dreamshaper_v7-int8-ov \
  -F prompt='a glowing crystal orb, vivid colors' \
  -F image=@/tmp/ovinfer_input.png \
  -F mask=@/tmp/ovinfer_mask.png \
  -F num_inference_steps=4 \
  -F strength=0.8 \
  | python -c 'import sys,json,base64; d=json.load(sys.stdin); b=base64.b64decode(d["data"][0]["b64_json"]); open("/tmp/ovinfer_inpaint.png","wb").write(b); print("created=",d["created"],"bytes=",len(b))'
file /tmp/ovinfer_inpaint.png
```

Expected: `file` reports `PNG image data` (512 x 512). Non-zero byte count. (Open the PNG to confirm the change is concentrated in the masked center region.)

- [ ] **Step 4: Verify a grayscale (no-alpha) mask also works**

```bash
.venv/bin/python - <<'PY'
from PIL import Image, ImageDraw
m = Image.new('L', (512, 512), 0)          # black = keep
d = ImageDraw.Draw(m)
d.ellipse([160, 160, 352, 352], fill=255)  # white = edit
m.save('/tmp/ovinfer_mask_gray.png')
PY
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8013/v1/images/edits \
  -F model=OpenVINO/LCM_Dreamshaper_v7-int8-ov \
  -F prompt='a teacup' \
  -F image=@/tmp/ovinfer_input.png \
  -F mask=@/tmp/ovinfer_mask_gray.png \
  -F num_inference_steps=4
```

Expected: `200`.

- [ ] **Step 5: Verify no-mask still routes to Image2Image (200)**

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8013/v1/images/edits \
  -F model=OpenVINO/LCM_Dreamshaper_v7-int8-ov \
  -F prompt='a teacup' \
  -F image=@/tmp/ovinfer_input.png \
  -F num_inference_steps=4 \
  -F strength=0.6
```

Expected: `200` (Image2Image path unchanged).

- [ ] **Step 6: Verify a non-diffusion model + mask is now 400 (not 501)**

```bash
curl -s -o /dev/null -w '%{http_code}\n' http://127.0.0.1:8013/v1/images/edits \
  -F model=OpenVINO/Qwen3-0.6B-int4-ov \
  -F prompt='x' \
  -F image=@/tmp/ovinfer_input.png \
  -F mask=@/tmp/ovinfer_mask.png
```

Expected: `400`. (Substitute any local plain-LLM model id from `/v1/models` if that one is absent. This confirms the ordering change: a mask no longer wins a 501 over the diffusion check.)

- [ ] **Step 7: Stop the server**

Stop the uvicorn process (Ctrl-C or kill the background job).

- [ ] **Step 8: Finalize the branch**

Use the `superpowers:finishing-a-development-branch` skill to decide merge/PR/cleanup for `feat/inpainting-pipeline`.

---

## Self-Review (completed during planning)

- **Spec coverage** (against `2026-06-12-inpainting-design.md`):
  - og facts / `generate(prompt, image, mask_image, **kwargs)` + `strength` — Task 3.
  - Hybrid mask semantics (`_decode_mask_tensor`: alpha inverted; else luminance; `(1,H,W,3)`) — Task 2.
  - Route fixes intent; mask split in backend (non-empty → Inpainting, empty → Image2Image) — Task 3.
  - Param mapping (only-set kwargs, `0.0` passthrough for `guidance_scale`/`strength`) — Task 3.
  - `_create_pipeline` `inpainting` branch — Task 3.
  - `image_edit` check order (404 → 400 unsupported → 400 invalid image → 400 invalid mask → build), diffusion check moved above the mask split — Task 3.
  - Transport unchanged; docstring `(implemented)`; mask→200 contract locked — Task 4.
  - **Behavior change consequence** (501-over-400 ordering removed): two dispatch tests updated in Task 3, one transport test removed in Task 4 — covered.
  - Errors: all four already in `_ERROR_MAP`, no `errors.py` change — confirmed in File Structure (no task needed).
  - Config `PIPELINE_KINDS` + preload — Task 1.
  - NPU: `reject_reason` falls through to `None` for `"inpainting"`, no `npu.py` change — confirmed in File Structure (no task needed).
  - Edge cases (empty mask → Image2Image, no-alpha luminance, alpha inversion, mask/image size mismatch → og 500, 0.0 knobs, seed default vs set, url vs b64, n>1) — covered across Task 2/3 tests + Task 6 acceptance (alpha + grayscale + no-mask + non-diffusion).
  - Acceptance on `LCM_Dreamshaper_v7-int8-ov` (has `vae_encoder`), spare port via `openvino-env`, RGBA + grayscale + no-mask + non-diffusion(400) checks — Task 6.
  - State docs (+ stale Image2Image mask note fixed) — Task 5.
  No gaps.
- **Placeholders:** none — every code step shows complete code and exact commands.
- **Type consistency:** `image_edit(model_id, prompt, image: bytes, mask: bytes | None, *, n, width, height, num_inference_steps, guidance_scale, negative_prompt, strength, seed) -> ImageResult` (signature unchanged from the merged Image2Image version); `_decode_mask_tensor(raw: bytes) -> ov.Tensor` (4D NHWC); `_decode_image_tensor(raw, batched=True) -> ov.Tensor` (reused); `ImageResult.images: list[bytes]`; `_encode_pngs(tensor) -> list[bytes]`. The `_FakeInpaintPipe.generate(prompt, image, mask_image, **kwargs)` signature matches og and the backend's masked-branch call; the transport `_StubEditBackend.image_edit` signature matches the real one. Consistent across tasks.
```
