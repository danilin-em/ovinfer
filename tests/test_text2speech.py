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
