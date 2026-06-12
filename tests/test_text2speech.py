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
