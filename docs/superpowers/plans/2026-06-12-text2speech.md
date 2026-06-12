# Text2Speech (TTS) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Fill the existing `backend.speech` stub so `POST /v1/audio/speech` synthesizes speech with `og.Text2SpeechPipeline` (SpeechT5) and returns audio in wav/pcm/mp3/opus/aac/flac.

**Architecture:** "Inverted Whisper" — text in, samples out. The sync core (`Backend.speech`) resolves the model (404), checks the TTS marker (400), runs one blocking `generate(text)`, and returns transport-agnostic 16 kHz float32 samples in a `SpeechResult`. The transport encodes those samples to the requested container/codec (`_encode_speech`) and sets the media type. Same `resolve→type-check→build→generate` pipeline and "wire format is a transport detail" split as Whisper.

**Tech Stack:** openvino-genai (`og.Text2SpeechPipeline`), FastAPI, PyAV (`av`, already a dependency) for mp3/opus/aac/flac, stdlib `wave` for wav/pcm, numpy. Tests: `unittest` (no pytest in this project; `og` is never imported — dispatch stubs `_pipeline`, transport stubs the backend).

**Spec:** `docs/superpowers/specs/2026-06-12-text2speech-design.md`
**Branch:** `feat/text2speech-pipeline` (already checked out; spec already committed).

---

## File Structure

- **Modify** `ovinfer/config.py` — add `"tts"` to `PIPELINE_KINDS`.
- **Modify** `ovinfer/registry.py` — add `is_tts_model(model_dir)` (marker: `openvino_vocoder.xml`).
- **Modify** `ovinfer/backend.py` — add `_TTS_SR`, `SpeechResult` dataclass, `_speech_samples` helper; fill the `speech()` stub; add the `"tts"` branch in `_create_pipeline`.
- **Modify** `ovinfer/server.py` — add `_TTS_MEDIA_TYPES`, `_TTS_PYAV_CODECS`, `_to_int16`, `_encode_speech`; change `SpeechRequest.response_format` to a `Literal` defaulting to `"mp3"`; rewrite the `audio_speech` route; update the module-docstring pipeline map.
- **Create** `tests/test_text2speech.py` — unit (kind, marker, sample extraction, encode), dispatch (stub `_pipeline`), contract (`TestClient` + stub backend). Built up incrementally; imports placed right before the class that needs them (matches `tests/test_whisper.py` style), so the file stays runnable after each task.

**Test commands:**
- One class: `.venv/bin/python -m unittest tests.test_text2speech.ClassName -v`
- Whole file: `.venv/bin/python -m unittest tests.test_text2speech -v`
- Full suite: `.venv/bin/python -m unittest discover -s tests`

Baseline before starting: full suite is green (167 tests).

---

## Task 1: `tts` pipeline kind

**Files:**
- Modify: `ovinfer/config.py:14-23`
- Test: `tests/test_text2speech.py` (create)

- [ ] **Step 1: Write the failing test** — create `tests/test_text2speech.py`:

```python
"""Tests for Text2Speech (TTS) support: the tts pipeline kind, the is_tts_model
marker, sample extraction from og results, the audio encoders (wav/pcm/mp3/opus/
aac/flac), Backend.speech (dispatch), and POST /v1/audio/speech.

og (openvino_genai) is never imported for assertions: pure helpers are tested
directly, dispatch tests stub _pipeline, transport tests stub the whole backend.
av/numpy/openvino are plain deps and may be imported.
"""

import unittest

from ovinfer import config


class PipelineKindTest(unittest.TestCase):
    def test_tts_is_a_known_kind(self):
        self.assertIn("tts", config.PIPELINE_KINDS)

    def test_preload_accepts_tts(self):
        self.assertEqual(config.parse_preload("org/m:tts"), [("org/m", "tts")])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests.test_text2speech.PipelineKindTest -v`
Expected: FAIL — `test_tts_is_a_known_kind` AssertionError (`'tts' not found`), `test_preload_accepts_tts` raises `ConfigError`.

- [ ] **Step 3: Write minimal implementation** — in `ovinfer/config.py`, add `"tts"` to the `PIPELINE_KINDS` tuple:

```python
PIPELINE_KINDS = (
    "llm",
    "vlm",
    "embedding",
    "rerank",
    "text2image",
    "image2image",
    "inpainting",
    "whisper",
    "tts",
)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m unittest tests.test_text2speech.PipelineKindTest -v`
Expected: PASS (2 tests).

- [ ] **Step 5: Commit**

```bash
git add ovinfer/config.py tests/test_text2speech.py
git commit -m "feat: register the tts pipeline kind"
```

---

## Task 2: `is_tts_model` registry marker

**Files:**
- Modify: `ovinfer/registry.py` (append after `is_whisper_model`)
- Test: `tests/test_text2speech.py` (append)

- [ ] **Step 1: Write the failing test** — append to `tests/test_text2speech.py`:

```python
import tempfile
from pathlib import Path

from ovinfer import registry


def _tts_root(model: str = "org/tts") -> tuple[Path, str]:
    """Temp models_root holding a SpeechT5 export (has openvino_vocoder.xml,
    and — like every encoder-decoder export — openvino_encoder_model.xml too)."""
    root = Path(tempfile.mkdtemp())
    d = root / model
    d.mkdir(parents=True)
    (d / "openvino_encoder_model.xml").write_text("<net/>")
    (d / "openvino_decoder_model.xml").write_text("<net/>")
    (d / "openvino_postnet.xml").write_text("<net/>")
    (d / "openvino_vocoder.xml").write_text("<net/>")
    return root, model


def _whisper_root(model: str = "org/whisper") -> tuple[Path, str]:
    """Temp models_root holding a Whisper export (encoder, no vocoder)."""
    root = Path(tempfile.mkdtemp())
    d = root / model
    d.mkdir(parents=True)
    (d / "openvino_encoder_model.xml").write_text("<net/>")
    (d / "openvino_decoder_model.xml").write_text("<net/>")
    return root, model


def _llm_root(model: str = "org/llm") -> tuple[Path, str]:
    """Temp models_root holding a plain-LLM export (no encoder, no vocoder)."""
    root = Path(tempfile.mkdtemp())
    d = root / model
    d.mkdir(parents=True)
    (d / "openvino_model.xml").write_text("<net/>")
    return root, model


class IsTtsModelTest(unittest.TestCase):
    def test_true_when_vocoder_present(self):
        root, model = _tts_root()
        self.assertTrue(registry.is_tts_model(root / model))

    def test_false_for_whisper_export(self):
        # Whisper has an encoder but no vocoder -> not a TTS model (clean 400).
        root, model = _whisper_root()
        self.assertFalse(registry.is_tts_model(root / model))

    def test_false_for_plain_llm(self):
        root, model = _llm_root()
        self.assertFalse(registry.is_tts_model(root / model))

    def test_false_for_nonexistent_path(self):
        self.assertFalse(registry.is_tts_model(Path("/no/such/dir")))
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests.test_text2speech.IsTtsModelTest -v`
Expected: FAIL — `AttributeError: module 'ovinfer.registry' has no attribute 'is_tts_model'`.

- [ ] **Step 3: Write minimal implementation** — append to `ovinfer/registry.py`:

```python
def is_tts_model(model_dir: Path) -> bool:
    """True if the model dir holds a Text2Speech (SpeechT5) export.

    SpeechT5 exports ship openvino_vocoder.xml (a separate neural vocoder)
    alongside encoder/decoder/postnet; LLM, VLM, diffusion and Whisper exports
    do not. POST /v1/audio/speech requires this marker; a model without it is
    rejected with 400 rather than failing pipeline load.

    A SpeechT5 export also carries openvino_encoder_model.xml (it is an
    encoder-decoder model), so is_whisper_model() also matches it; dispatch is
    disambiguated by endpoint, not by marker. og.Text2SpeechPipeline supports
    only SpeechT5 — Kokoro/OuteTTS/Qwen-TTS exports lack a vocoder and are
    rejected here (400), like a non-SD diffusion model at image generation.

    Returns False (not an error) for a nonexistent path.
    """
    return (model_dir / "openvino_vocoder.xml").is_file()
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m unittest tests.test_text2speech.IsTtsModelTest -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add ovinfer/registry.py tests/test_text2speech.py
git commit -m "feat: add is_tts_model registry marker (openvino_vocoder.xml)"
```

---

## Task 3: `SpeechResult` + `_speech_samples` (sample extraction)

**Files:**
- Modify: `ovinfer/backend.py` — add `_TTS_SR`, `SpeechResult`, `_speech_samples`
- Test: `tests/test_text2speech.py` (append)

- [ ] **Step 1: Write the failing test** — append to `tests/test_text2speech.py`:

```python
import numpy as np
import openvino as ov

from ovinfer.backend import _TTS_SR, SpeechResult, _speech_samples


class _FakeSpeechResult:
    """Stands in for og.Text2SpeechDecodedResults: .speeches is a list of tensors."""

    def __init__(self, speeches):
        self.speeches = speeches


class SpeechSamplesTest(unittest.TestCase):
    def test_sample_rate_is_16k(self):
        self.assertEqual(_TTS_SR, 16000)

    def test_extracts_first_speech_as_float32_1d_from_ov_tensor(self):
        tensor = ov.Tensor(np.array([[0.1, -0.2, 0.3]], dtype=np.float32))
        out = _speech_samples(_FakeSpeechResult([tensor]))
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out.ndim, 1)
        np.testing.assert_allclose(out, [0.1, -0.2, 0.3], rtol=1e-6)

    def test_accepts_plain_ndarray_and_flattens(self):
        out = _speech_samples(
            _FakeSpeechResult([np.array([[0.0, 0.5], [-0.5, 1.0]], dtype=np.float32)])
        )
        self.assertEqual(out.ndim, 1)
        self.assertEqual(len(out), 4)

    def test_speech_result_is_a_frozen_dataclass(self):
        r = SpeechResult(samples=np.zeros(3, dtype=np.float32), sample_rate=16000)
        self.assertEqual(r.sample_rate, 16000)
        with self.assertRaises(Exception):
            r.sample_rate = 8000  # frozen
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests.test_text2speech.SpeechSamplesTest -v`
Expected: FAIL — `ImportError: cannot import name '_TTS_SR' from 'ovinfer.backend'`.

- [ ] **Step 3: Write minimal implementation** — in `ovinfer/backend.py`:

(a) Add the `SpeechResult` dataclass right after the `TranscriptionResult` dataclass (near line 80):

```python
@dataclass(frozen=True)
class SpeechResult:
    samples: Any      # numpy float32, mono, ~[-1, 1]
    sample_rate: int  # 16000 (SpeechT5)
```

(b) Add `_TTS_SR` and `_speech_samples` near the audio helpers (after `_decode_audio`, around line 293):

```python
_TTS_SR = 16000


def _speech_samples(result: Any) -> Any:
    """Extract a 1-D float32 waveform from a Text2SpeechDecodedResults.

    og returns .speeches as a list of ov.Tensor (one per input text); we send a
    single text, so speeches[0] is the waveform. Like _encode_pngs, the tensor's
    backing numpy view (.data) is taken explicitly (np.array on an ov.Tensor
    wraps it as a 0-d object array). Shape ((1, n) vs (n,)) is flattened with
    reshape(-1).
    """
    import numpy as np
    import openvino as ov

    speech = result.speeches[0]
    arr = np.asarray(speech.data if isinstance(speech, ov.Tensor) else speech)
    return arr.reshape(-1).astype(np.float32)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m unittest tests.test_text2speech.SpeechSamplesTest -v`
Expected: PASS (4 tests).

- [ ] **Step 5: Commit**

```bash
git add ovinfer/backend.py tests/test_text2speech.py
git commit -m "feat: add SpeechResult and _speech_samples sample extraction"
```

---

## Task 4: `Backend.speech` dispatch + pipeline branch

**Files:**
- Modify: `ovinfer/backend.py` — fill the `speech()` stub; add `"tts"` branch in `_create_pipeline`
- Test: `tests/test_text2speech.py` (append)

- [ ] **Step 1: Write the failing test** — append to `tests/test_text2speech.py`:

```python
from contextlib import contextmanager

from ovinfer.backend import Backend
from ovinfer.errors import ModelNotFound, UnsupportedModel


class _FakeTtsPipe:
    """Records generate(text, **kwargs); returns a fixed fake result."""

    def __init__(self):
        self.calls: list[dict] = []

    def generate(self, text, **kwargs):
        self.calls.append({"text": text, "kwargs": kwargs})
        return _FakeSpeechResult([np.array([0.1, 0.2, 0.3], dtype=np.float32)])


class _DispatchBackend(Backend):
    """Real Backend with _pipeline stubbed; records (model_id, kind) requests."""

    def __init__(self, models_root: Path, pipe: _FakeTtsPipe) -> None:
        super().__init__(models_root=models_root, default_device="CPU")
        self._pipe = pipe
        self.pipeline_requests: list[tuple[str, str]] = []

    @contextmanager
    def _pipeline(self, model_id, kind):
        self.pipeline_requests.append((model_id, kind))
        yield self._pipe


class SpeechDispatchTest(unittest.TestCase):
    def test_requests_tts_kind_and_returns_samples(self):
        root, model = _tts_root()
        pipe = _FakeTtsPipe()
        be = _DispatchBackend(root, pipe)
        result = be.speech(model, "hello world")
        self.assertIsInstance(result, SpeechResult)
        self.assertEqual(result.sample_rate, 16000)
        self.assertEqual(result.samples.ndim, 1)
        self.assertEqual(be.pipeline_requests, [(model, "tts")])

    def test_generate_called_with_text_only(self):
        root, model = _tts_root()
        pipe = _FakeTtsPipe()
        be = _DispatchBackend(root, pipe)
        be.speech(model, "hello world")
        self.assertEqual(pipe.calls[0]["text"], "hello world")
        # voice is not forwarded to og: SpeechT5 uses the built-in default speaker.
        self.assertEqual(pipe.calls[0]["kwargs"], {})

    def test_voice_is_accepted_but_ignored(self):
        root, model = _tts_root()
        pipe = _FakeTtsPipe()
        be = _DispatchBackend(root, pipe)
        be.speech(model, "hi", voice="alloy")
        self.assertEqual(pipe.calls[0]["kwargs"], {})

    def test_non_tts_model_is_unsupported_without_pipeline(self):
        root, model = _whisper_root()  # encoder but no vocoder
        be = _DispatchBackend(root, _FakeTtsPipe())
        with self.assertRaises(UnsupportedModel):
            be.speech(model, "hi")
        self.assertEqual(be.pipeline_requests, [])

    def test_unknown_model_404s_before_anything(self):
        root, _ = _tts_root()
        be = _DispatchBackend(root, _FakeTtsPipe())
        with self.assertRaises(ModelNotFound):
            be.speech("org/ghost", "hi")
        self.assertEqual(be.pipeline_requests, [])
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests.test_text2speech.SpeechDispatchTest -v`
Expected: FAIL — `test_requests_tts_kind_and_returns_samples` raises `PipelineNotImplemented` (current stub), and the others fail (stub never reaches the checks / returns).

- [ ] **Step 3: Write minimal implementation** — in `ovinfer/backend.py`:

(a) Replace the `speech` stub (around line 892) with:

```python
    def speech(self, model_id: str, text: str, voice: str | None = None) -> SpeechResult:
        """Text-to-speech via og.Text2SpeechPipeline (SpeechT5).

        Resolve (404) then the TTS-model check (400) run before the pipeline is
        built, matching _audio_generate. voice is accepted for OpenAI
        compatibility but ignored: SpeechT5 selects a voice via a speaker
        embedding, not a named voice, and og's built-in default speaker (baked
        into the runtime, offline) is used. Returns transport-agnostic 16 kHz
        float32 samples; the transport encodes them to the requested format.
        """
        model_dir = registry.resolve_model_dir(self.settings.models_root, model_id)
        if not registry.is_tts_model(model_dir):
            raise UnsupportedModel(
                f"model '{model_id}' is not a Text2Speech (SpeechT5) model "
                "and cannot synthesize speech"
            )
        with self._pipeline(model_id, "tts") as pipe:
            result = pipe.generate(text)
        return SpeechResult(samples=_speech_samples(result), sample_rate=_TTS_SR)
```

(b) Add the `"tts"` branch in `_create_pipeline`, right after the `whisper` branch (around line 449):

```python
        if kind == "tts":
            return og.Text2SpeechPipeline(model_dir, device)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m unittest tests.test_text2speech.SpeechDispatchTest -v`
Expected: PASS (5 tests).

- [ ] **Step 5: Commit**

```bash
git add ovinfer/backend.py tests/test_text2speech.py
git commit -m "feat: implement Backend.speech dispatch (SpeechT5, default speaker)"
```

---

## Task 5: `_encode_speech` audio encoders

**Files:**
- Modify: `ovinfer/server.py` — add `_TTS_MEDIA_TYPES`, `_TTS_PYAV_CODECS`, `_to_int16`, `_encode_speech`
- Test: `tests/test_text2speech.py` (append)

> The PyAV plumbing below (s16 AudioFrame → `AudioResampler(format=stream.format.name)` → encode → mux → flush) was verified to produce decodable mp3/opus/aac/flac at 16 kHz mono. opus/aac change the decoded sample count (internal resample / encoder priming), so the round-trip test asserts *decodability*, not an exact sample count.

- [ ] **Step 1: Write the failing test** — append to `tests/test_text2speech.py`:

```python
from ovinfer.server import _TTS_MEDIA_TYPES, _encode_speech


def _sine(seconds: float = 0.3, sr: int = 16000):
    n = int(seconds * sr)
    t = np.arange(n, dtype=np.float32)
    return (0.2 * np.sin(2 * np.pi * 440 * t / sr)).astype(np.float32), sr


def _decode_count(data: bytes) -> int:
    """Total decoded sample count via PyAV (proves the bytes are valid audio)."""
    import io

    import av

    container = av.open(io.BytesIO(data))
    total = sum(frame.samples for frame in container.decode(container.streams.audio[0]))
    container.close()
    return total


class EncodeSpeechTest(unittest.TestCase):
    def test_media_types_cover_all_formats(self):
        self.assertEqual(
            set(_TTS_MEDIA_TYPES),
            {"wav", "pcm", "mp3", "opus", "aac", "flac"},
        )

    def test_pcm_is_raw_int16_no_header(self):
        s, sr = _sine()
        out = _encode_speech(s, sr, "pcm")
        self.assertEqual(len(out), 2 * len(s))  # 16-bit mono, no header

    def test_wav_has_riff_header_and_decodes(self):
        s, sr = _sine()
        out = _encode_speech(s, sr, "wav")
        self.assertEqual(out[:4], b"RIFF")
        self.assertEqual(_decode_count(out), len(s))  # lossless, exact

    def test_flac_has_magic_and_decodes(self):
        s, sr = _sine()
        out = _encode_speech(s, sr, "flac")
        self.assertEqual(out[:4], b"fLaC")
        self.assertEqual(_decode_count(out), len(s))  # lossless, exact

    def test_opus_has_ogg_magic_and_decodes(self):
        s, sr = _sine()
        out = _encode_speech(s, sr, "opus")
        self.assertEqual(out[:4], b"OggS")
        self.assertGreater(_decode_count(out), 0)

    def test_mp3_decodes_and_starts_plausibly(self):
        s, sr = _sine()
        out = _encode_speech(s, sr, "mp3")
        self.assertTrue(out.startswith(b"ID3") or out[0] == 0xFF, out[:4])
        self.assertGreater(_decode_count(out), 0)

    def test_aac_adts_sync_and_decodes(self):
        s, sr = _sine()
        out = _encode_speech(s, sr, "aac")
        self.assertEqual(out[0], 0xFF)            # ADTS sync word (0xFFF...)
        self.assertEqual(out[1] & 0xF0, 0xF0)
        self.assertGreater(_decode_count(out), 0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests.test_text2speech.EncodeSpeechTest -v`
Expected: FAIL — `ImportError: cannot import name '_TTS_MEDIA_TYPES' from 'ovinfer.server'`.

- [ ] **Step 3: Write minimal implementation** — in `ovinfer/server.py`, add an audio-speech rendering section (place it after the audio-transcription rendering helpers, before `create_app`):

```python
# --- audio speech encoding ------------------------------------------------------

_TTS_MEDIA_TYPES = {
    "wav": "audio/wav",
    "pcm": "audio/pcm",
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
}

# (container format, codec) for the PyAV-encoded formats. wav/pcm use stdlib.
_TTS_PYAV_CODECS = {
    "mp3": ("mp3", "libmp3lame"),
    "opus": ("ogg", "libopus"),
    "aac": ("adts", "aac"),
    "flac": ("flac", "flac"),
}


def _to_int16(samples: Any) -> Any:
    """Float32 waveform in ~[-1, 1] -> clipped little-endian int16 PCM."""
    import numpy as np

    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")


def _encode_speech(samples: Any, sample_rate: int, response_format: str) -> bytes:
    """Encode a mono float32 waveform to the requested container/codec.

    wav/pcm use stdlib: float is scaled to int16 PCM; wav adds a RIFF header,
    pcm is the raw little-endian int16 bytes with no header. mp3/opus/aac/flac
    are muxed in-memory via PyAV (already a dependency): an AudioResampler
    converts the s16 frame to the codec's sample format, frames are encoded and
    the packets muxed, then the encoder is flushed.
    """
    import io
    import wave

    pcm = _to_int16(samples)
    if response_format == "pcm":
        return pcm.tobytes()
    if response_format == "wav":
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(pcm.tobytes())
        return buf.getvalue()

    import av

    container_format, codec = _TTS_PYAV_CODECS[response_format]
    buf = io.BytesIO()
    container = av.open(buf, mode="w", format=container_format)
    try:
        stream = container.add_stream(codec, rate=sample_rate)
        frame = av.AudioFrame.from_ndarray(
            pcm.reshape(1, -1), format="s16", layout="mono"
        )
        frame.sample_rate = sample_rate
        resampler = av.AudioResampler(
            format=stream.format.name, layout="mono", rate=sample_rate
        )
        for rframe in resampler.resample(frame):
            for packet in stream.encode(rframe):
                container.mux(packet)
        for packet in stream.encode(None):  # flush the encoder
            container.mux(packet)
    finally:
        container.close()
    return buf.getvalue()
```

(`Any` is already imported in `server.py` via `from typing import Any, Iterator, Literal`.)

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m unittest tests.test_text2speech.EncodeSpeechTest -v`
Expected: PASS (7 tests).

- [ ] **Step 5: Commit**

```bash
git add ovinfer/server.py tests/test_text2speech.py
git commit -m "feat: add _encode_speech (wav/pcm stdlib, mp3/opus/aac/flac via PyAV)"
```

---

## Task 6: `SpeechRequest` schema + `audio_speech` route

**Files:**
- Modify: `ovinfer/server.py` — `SpeechRequest.response_format` (`Literal`, default `"mp3"`); rewrite `audio_speech`; update docstring map (`server.py:18`)
- Test: `tests/test_text2speech.py` (append)

- [ ] **Step 1: Write the failing test** — append to `tests/test_text2speech.py`:

```python
from fastapi.testclient import TestClient

from ovinfer.server import create_app

MODEL = "org/tts"


class _StubSpeechBackend:
    """Duck-typed Backend for transport tests; records speech() calls."""

    def __init__(self, result_or_exc) -> None:
        self.result_or_exc = result_or_exc
        self.calls: list[dict] = []

    def speech(self, model_id, text, voice=None):
        self.calls.append({"model_id": model_id, "text": text, "voice": voice})
        if isinstance(self.result_or_exc, Exception):
            raise self.result_or_exc
        return self.result_or_exc


def _speech_client(result_or_exc):
    backend = _StubSpeechBackend(result_or_exc)
    return TestClient(create_app(backend)), backend


def _ok_speech():
    s, sr = _sine(seconds=0.2)
    return SpeechResult(samples=s, sample_rate=sr)


class SpeechRouteTest(unittest.TestCase):
    def test_default_format_is_mp3(self):
        client, backend = _speech_client(_ok_speech())
        resp = client.post("/v1/audio/speech", json={"model": MODEL, "input": "hello"})
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.headers["content-type"], "audio/mpeg")
        self.assertGreater(len(resp.content), 0)
        self.assertEqual(backend.calls[0]["text"], "hello")

    def test_each_format_sets_content_type(self):
        for fmt, ct in [
            ("wav", "audio/wav"), ("pcm", "audio/pcm"), ("mp3", "audio/mpeg"),
            ("opus", "audio/ogg"), ("aac", "audio/aac"), ("flac", "audio/flac"),
        ]:
            client, _ = _speech_client(_ok_speech())
            resp = client.post(
                "/v1/audio/speech",
                json={"model": MODEL, "input": "hi", "response_format": fmt},
            )
            self.assertEqual(resp.status_code, 200, fmt)
            self.assertEqual(resp.headers["content-type"], ct, fmt)
            self.assertGreater(len(resp.content), 0, fmt)

    def test_invalid_format_is_422(self):
        client, _ = _speech_client(_ok_speech())
        resp = client.post(
            "/v1/audio/speech",
            json={"model": MODEL, "input": "hi", "response_format": "bogus"},
        )
        self.assertEqual(resp.status_code, 422)

    def test_voice_is_forwarded_to_backend(self):
        client, backend = _speech_client(_ok_speech())
        client.post(
            "/v1/audio/speech",
            json={"model": MODEL, "input": "hi", "voice": "alloy"},
        )
        self.assertEqual(backend.calls[0]["voice"], "alloy")

    def test_unsupported_model_maps_to_400(self):
        client, _ = _speech_client(UnsupportedModel("not a tts model"))
        resp = client.post("/v1/audio/speech", json={"model": MODEL, "input": "hi"})
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["error"]["code"], "unsupported_model")

    def test_model_not_found_maps_to_404(self):
        client, _ = _speech_client(ModelNotFound("org/ghost", "/models"))
        resp = client.post("/v1/audio/speech", json={"model": "org/ghost", "input": "hi"})
        self.assertEqual(resp.status_code, 404)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `.venv/bin/python -m unittest tests.test_text2speech.SpeechRouteTest -v`
Expected: FAIL — `test_default_format_is_mp3` gets `audio/wav` (current default) and `test_invalid_format_is_422` gets 200 (current `response_format: str` accepts anything), among others.

- [ ] **Step 3: Write minimal implementation** — in `ovinfer/server.py`:

(a) Replace the `SpeechRequest` model (around line 120):

```python
class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    input: str
    # Accepted for OpenAI compatibility but ignored: SpeechT5 selects a voice via
    # a speaker embedding, not a named voice; og's default speaker is used.
    voice: str | None = None
    # SpeechT5 outputs 16 kHz audio; the transport encodes it to the requested
    # container/codec. Default mp3 matches OpenAI clients' default.
    response_format: Literal["wav", "pcm", "mp3", "opus", "aac", "flac"] = "mp3"
```

(b) Replace the `audio_speech` route (around line 544):

```python
    @app.post("/v1/audio/speech")
    def audio_speech(req: SpeechRequest) -> Response:
        result = backend.speech(req.model, req.input, req.voice)
        audio = _encode_speech(result.samples, result.sample_rate, req.response_format)
        return Response(
            content=audio, media_type=_TTS_MEDIA_TYPES[req.response_format]
        )
```

(c) Update the module-docstring pipeline map (`server.py:18`):

```python
    Text2SpeechPipeline        -> POST /v1/audio/speech              (implemented)
```

- [ ] **Step 4: Run test to verify it passes**

Run: `.venv/bin/python -m unittest tests.test_text2speech.SpeechRouteTest -v`
Expected: PASS (6 tests).

- [ ] **Step 5: Run the whole new file + full suite**

Run: `.venv/bin/python -m unittest tests.test_text2speech -v`
Expected: PASS (all classes).
Run: `.venv/bin/python -m unittest discover -s tests`
Expected: OK, count increased from 167 by the new tests.

- [ ] **Step 6: Commit**

```bash
git add ovinfer/server.py tests/test_text2speech.py
git commit -m "feat: wire POST /v1/audio/speech (six formats, default mp3)"
```

---

## Task 7: Record project state

**Files:**
- Modify: `CLAUDE.md` — add a TTS entry to the "Состояние" section; update "Следующие шаги" if needed.

- [ ] **Step 1: Update `CLAUDE.md`** — in the "## Состояние" section, bump the date line to `2026-06-12` and add a bullet after the Whisper entry (write in the same style/Russian as the surrounding entries):

```markdown
- **Text2Speech (TTS) реализован 2026-06-12**: `og.Text2SpeechPipeline` (SpeechT5)
  на `POST /v1/audio/speech` — заполнил стаб `backend.speech`. Резолв (404) →
  маркер `is_tts_model` (`openvino_vocoder.xml`, 400 non-tts) → `generate(text)`
  → 16 кГц float32 samples в `SpeechResult`; кодирование в транспорте
  (`_encode_speech`): wav/pcm — stdlib `wave`, mp3/opus/aac/flac — PyAV (уже в
  deps). `response_format` — `Literal` из шести, **дефолт mp3** (как у
  OpenAI-клиентов). `voice` принимается, но игнорируется (вшитый дефолтный
  speaker SpeechT5, офлайн); реальный выбор голоса — follow-up. SpeechT5 несёт и
  `openvino_encoder_model.xml` (маркер Whisper): диспетчеризация по эндпоинту,
  TTS→/transcriptions даёт build-500 (общий follow-up). Только SpeechT5 (как
  Text2Image — только SD/Flux/LCM); Kokoro/Qwen-TTS → 400. Приёмка end-to-end на
  `llmware/speech-t5-tts-ov` (CPU). Спека:
  `docs/superpowers/specs/2026-06-12-text2speech-design.md`; план:
  `docs/superpowers/plans/2026-06-12-text2speech.md`.
```

- [ ] **Step 2: Commit**

```bash
git add CLAUDE.md
git commit -m "docs: record Text2Speech (TTS) implementation in project state"
```

---

## Task 8: Acceptance on the real model (manual)

Not a unit test — this exercises the real pipeline through HTTP. `og` loads here, so the server runs **through `openvino-env`** (per the user's global rule). The acceptance model `llmware/speech-t5-tts-ov` is already downloaded; the marker `openvino_vocoder.xml` is confirmed present. The user keeps a server on `:8000`, so use a **spare port** (8011). Loopback curl needs the Bash sandbox disabled.

- [ ] **Step 1: Start the server (background) through openvino-env on a spare port**

```bash
OVINFER_MODELS_ROOT=~/.openvino/models/huggingface.co \
  openvino-env ovinfer server --listen 127.0.0.1:8011 --device CPU
```

- [ ] **Step 2: Sanity — model is listed**

```bash
curl -s http://127.0.0.1:8011/v1/models | python3 -c "import sys,json; print([m['id'] for m in json.load(sys.stdin)['data']])"
```
Expected: the list includes `llmware/speech-t5-tts-ov`.

- [ ] **Step 3: Synthesize each format and check magic bytes**

```bash
M=llmware/speech-t5-tts-ov
for fmt in mp3 wav pcm opus aac flac; do
  curl -s -X POST http://127.0.0.1:8011/v1/audio/speech \
    -H 'Content-Type: application/json' \
    -d "{\"model\":\"$M\",\"input\":\"OpenVINO text to speech is working.\",\"response_format\":\"$fmt\"}" \
    -o /tmp/tts_out.$fmt -D /tmp/tts_hdr.$fmt
  echo "=== $fmt ==="; grep -i content-type /tmp/tts_hdr.$fmt
  xxd -l 8 /tmp/tts_out.$fmt; ls -l /tmp/tts_out.$fmt | awk '{print $5" bytes"}'
done
```
Expected per format: 200, the Content-Type from `_TTS_MEDIA_TYPES`, non-empty body, and the magic bytes — `RIFF` (wav), `fLaC` (flac), `OggS` (opus), `ID3`/`0xFF…` (mp3), `0xFFF1` (aac), raw int16 (pcm). Optionally verify playback: `ffplay -autoexit /tmp/tts_out.mp3`.

- [ ] **Step 4: Default format is mp3 (no response_format)**

```bash
curl -s -X POST http://127.0.0.1:8011/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d "{\"model\":\"llmware/speech-t5-tts-ov\",\"input\":\"default format\"}" \
  -D - -o /tmp/tts_default.bin | grep -i content-type
```
Expected: `content-type: audio/mpeg`.

- [ ] **Step 5: Error paths**

```bash
# non-TTS model -> 400 unsupported_model
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:8011/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"OpenVINO/Qwen3-0.6B-int4-ov","input":"hi"}'   # expect 400
# unknown model -> 404
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:8011/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"org/ghost","input":"hi"}'                      # expect 404
# bad response_format -> 422
curl -s -o /dev/null -w "%{http_code}\n" -X POST http://127.0.0.1:8011/v1/audio/speech \
  -H 'Content-Type: application/json' \
  -d '{"model":"llmware/speech-t5-tts-ov","input":"hi","response_format":"bogus"}'  # expect 422
```

- [ ] **Step 6: Confirm spec assumptions (note any deviation)**

Verify and record: default speaker works **offline** (no network call during generate); `.speeches[0]` shape and that `_speech_samples` flattens it correctly (the wav byte length ≈ `2 × duration × 16000`); sample rate is 16 kHz. If `stream.format.name` or the `AudioFrame` plumbing needed adjustment for any codec, the round-trip unit test in Task 5 would already have caught it — reconcile here if real-model output differs.

- [ ] **Step 7: Stop the server**

```bash
pkill -f "ovinfer server --listen 127.0.0.1:8011" || true
```

---

## Self-Review (completed during planning)

- **Spec coverage:** marker (`is_tts_model`, T2) ✓; backend dispatch + default speaker + voice-ignored (T4) ✓; `_create_pipeline` branch (T4) ✓; six `response_format` + transport encoding (T5/T6) ✓; default mp3 + 422 (T6) ✓; errors 404/400 reuse `_ERROR_MAP` (T4/T6) ✓; config kind (T1) ✓; no new deps ✓; npu.py unchanged ✓ (no task touches it — correct); Whisper/TTS marker overlap documented (T2 docstring + CLAUDE.md) ✓; out-of-scope items (streaming, voice cloning, TTS knobs, chunking, non-SpeechT5) left unimplemented ✓.
- **Placeholders:** none — every step has runnable code/commands.
- **Type/name consistency:** `SpeechResult(samples, sample_rate)`, `_speech_samples`, `_TTS_SR`, `_encode_speech(samples, sample_rate, response_format)`, `_TTS_MEDIA_TYPES`, `_TTS_PYAV_CODECS`, `is_tts_model`, kind `"tts"` — used identically across tasks and tests.
```
