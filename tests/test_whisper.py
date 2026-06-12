"""Tests for Whisper (ASR) support: the whisper pipeline kind, the PyAV audio
decoder, Backend.transcribe/translate (dispatch/kwargs/result-mapping), the
response_format renderers, and POST /v1/audio/transcriptions|translations.

og (openvino_genai) is never imported for assertions: pure helpers are tested
directly, dispatch tests stub _pipeline, transport tests stub the whole backend.
av/numpy/PIL/openvino are plain deps and may be imported.
"""

import io
import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from ovinfer import config


class PipelineKindTest(unittest.TestCase):
    def test_whisper_is_a_known_kind(self):
        self.assertIn("whisper", config.PIPELINE_KINDS)

    def test_preload_accepts_whisper(self):
        self.assertEqual(
            config.parse_preload("org/m:whisper"), [("org/m", "whisper")]
        )


from ovinfer import registry


def _whisper_root(model: str = "org/whisper") -> tuple[Path, str]:
    """Temp models_root holding a Whisper export (has openvino_encoder_model.xml)."""
    root = Path(tempfile.mkdtemp())
    d = root / model
    d.mkdir(parents=True)
    (d / "openvino_encoder_model.xml").write_text("<net/>")
    (d / "openvino_decoder_model.xml").write_text("<net/>")
    return root, model


def _llm_root(model: str = "org/llm") -> tuple[Path, str]:
    """Temp models_root holding a plain-LLM export (no encoder, no model_index)."""
    root = Path(tempfile.mkdtemp())
    d = root / model
    d.mkdir(parents=True)
    (d / "openvino_model.xml").write_text("<net/>")
    return root, model


class IsWhisperModelTest(unittest.TestCase):
    def test_true_when_encoder_present(self):
        root, model = _whisper_root()
        self.assertTrue(registry.is_whisper_model(root / model))

    def test_false_for_plain_llm(self):
        root, model = _llm_root()
        self.assertFalse(registry.is_whisper_model(root / model))

    def test_false_for_nonexistent_path(self):
        self.assertFalse(registry.is_whisper_model(Path("/no/such/dir")))


import numpy as np

from ovinfer.backend import _TARGET_SR, _decode_audio
from ovinfer.errors import InvalidAudio


def _wav_bytes(seconds: float = 0.5, sr: int = 16000, channels: int = 1) -> bytes:
    """A valid mono/stereo 16-bit PCM WAV (stdlib only; PyAV reads it)."""
    import math
    import struct
    import wave

    n = int(seconds * sr)
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(sr)
        samples = []
        for t in range(n):
            val = int(0.2 * 32767 * math.sin(2 * math.pi * 440 * t / sr))
            samples.extend([val] * channels)
        w.writeframes(struct.pack("<%dh" % len(samples), *samples))
    return buf.getvalue()


class DecodeAudioTest(unittest.TestCase):
    def test_mono_16k_wav_decodes_to_float32_1d(self):
        out = _decode_audio(_wav_bytes(seconds=0.5, sr=16000))
        self.assertEqual(out.dtype, np.float32)
        self.assertEqual(out.ndim, 1)
        # 0.5 s at 16 kHz ~= 8000 samples (resampler may trim/pad a few).
        self.assertTrue(7000 <= len(out) <= 9000, len(out))
        self.assertLessEqual(float(np.abs(out).max()), 1.0)

    def test_target_sample_rate_is_16k(self):
        self.assertEqual(_TARGET_SR, 16000)

    def test_resamples_8k_up_to_16k(self):
        # 0.5 s at 8 kHz must come back as ~8000 samples (0.5 s at 16 kHz).
        out = _decode_audio(_wav_bytes(seconds=0.5, sr=8000))
        self.assertTrue(7000 <= len(out) <= 9000, len(out))

    def test_stereo_is_downmixed_to_mono_1d(self):
        out = _decode_audio(_wav_bytes(seconds=0.3, sr=16000, channels=2))
        self.assertEqual(out.ndim, 1)
        self.assertTrue(4000 <= len(out) <= 5600, len(out))

    def test_garbage_bytes_raise_invalid_audio(self):
        with self.assertRaises(InvalidAudio):
            _decode_audio(b"not audio at all")

    def test_empty_bytes_raise_invalid_audio(self):
        with self.assertRaises(InvalidAudio):
            _decode_audio(b"")

    def test_container_without_audio_stream_raises_invalid_audio(self):
        import av

        out = io.BytesIO()
        oc = av.open(out, mode="w", format="mp4")
        stream = oc.add_stream("mpeg4", rate=1)
        stream.width = 16
        stream.height = 16
        stream.pix_fmt = "yuv420p"
        frame = av.VideoFrame(16, 16, "yuv420p")
        for packet in stream.encode(frame):
            oc.mux(packet)
        for packet in stream.encode(None):  # flush
            oc.mux(packet)
        oc.close()
        with self.assertRaises(InvalidAudio):
            _decode_audio(out.getvalue())


if __name__ == "__main__":
    unittest.main()
