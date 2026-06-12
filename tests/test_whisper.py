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


from ovinfer.backend import Backend, Segment, TranscriptionResult
from ovinfer.errors import ModelNotFound, UnsupportedModel


class _FakeChunk:
    def __init__(self, start, end, text):
        self.start_ts, self.end_ts, self.text = start, end, text


class _FakeWhisperResult:
    """Stands in for og.WhisperDecodedResults."""

    texts = ["  hello world  "]
    language = "<|en|>"
    chunks = [_FakeChunk(0.0, 1.0, "hello"), _FakeChunk(1.0, 2.0, "world")]


class _FakeWhisperPipe:
    """Records generate(samples, **kwargs); returns a fixed fake result."""

    def __init__(self):
        self.calls: list[dict] = []

    def generate(self, samples, **kwargs):
        self.calls.append({"samples": samples, "kwargs": kwargs})
        return _FakeWhisperResult()


class _DispatchBackend(Backend):
    """Real Backend with _pipeline stubbed; records (model_id, kind) requests."""

    def __init__(self, models_root: Path, pipe: _FakeWhisperPipe) -> None:
        super().__init__(models_root=models_root, default_device="CPU")
        self._pipe = pipe
        self.pipeline_requests: list[tuple[str, str]] = []

    @contextmanager
    def _pipeline(self, model_id, kind):
        self.pipeline_requests.append((model_id, kind))
        yield self._pipe


class TranscribeDispatchTest(unittest.TestCase):
    def test_requests_whisper_kind_and_strips_text(self):
        root, model = _whisper_root()
        pipe = _FakeWhisperPipe()
        be = _DispatchBackend(root, pipe)
        result = be.transcribe(model, _wav_bytes())
        self.assertIsInstance(result, TranscriptionResult)
        self.assertEqual(result.text, "hello world")  # stripped
        self.assertEqual(result.language, "en")        # <|en|> -> en
        self.assertIsNone(result.segments)             # json -> no timestamps
        self.assertGreater(result.duration, 0.0)
        self.assertEqual(be.pipeline_requests, [(model, "whisper")])

    def test_transcribe_task_and_no_optional_kwargs(self):
        root, model = _whisper_root()
        pipe = _FakeWhisperPipe()
        be = _DispatchBackend(root, pipe)
        be.transcribe(model, _wav_bytes())
        # Only task is set; language/initial_prompt/do_sample/return_timestamps absent.
        self.assertEqual(pipe.calls[0]["kwargs"], {"task": "transcribe"})

    def test_translate_sets_translate_task(self):
        root, model = _whisper_root()
        pipe = _FakeWhisperPipe()
        be = _DispatchBackend(root, pipe)
        be.translate(model, _wav_bytes())
        self.assertEqual(pipe.calls[0]["kwargs"], {"task": "translate"})

    def test_language_is_wrapped_in_tokens(self):
        root, model = _whisper_root()
        pipe = _FakeWhisperPipe()
        be = _DispatchBackend(root, pipe)
        be.transcribe(model, _wav_bytes(), language="ru")
        self.assertEqual(pipe.calls[0]["kwargs"]["language"], "<|ru|>")

    def test_prompt_becomes_initial_prompt(self):
        root, model = _whisper_root()
        pipe = _FakeWhisperPipe()
        be = _DispatchBackend(root, pipe)
        be.transcribe(model, _wav_bytes(), prompt="hi there")
        self.assertEqual(pipe.calls[0]["kwargs"]["initial_prompt"], "hi there")

    def test_temperature_above_zero_enables_sampling(self):
        root, model = _whisper_root()
        pipe = _FakeWhisperPipe()
        be = _DispatchBackend(root, pipe)
        be.transcribe(model, _wav_bytes(), temperature=0.7)
        self.assertEqual(pipe.calls[0]["kwargs"]["do_sample"], True)
        self.assertEqual(pipe.calls[0]["kwargs"]["temperature"], 0.7)

    def test_temperature_zero_stays_greedy(self):
        root, model = _whisper_root()
        pipe = _FakeWhisperPipe()
        be = _DispatchBackend(root, pipe)
        be.transcribe(model, _wav_bytes(), temperature=0.0)
        self.assertNotIn("do_sample", pipe.calls[0]["kwargs"])
        self.assertNotIn("temperature", pipe.calls[0]["kwargs"])

    def test_timestamp_format_requests_chunks_and_maps_segments(self):
        root, model = _whisper_root()
        pipe = _FakeWhisperPipe()
        be = _DispatchBackend(root, pipe)
        result = be.transcribe(model, _wav_bytes(), response_format="srt")
        self.assertEqual(pipe.calls[0]["kwargs"]["return_timestamps"], True)
        self.assertEqual(
            result.segments,
            [Segment(0.0, 1.0, "hello"), Segment(1.0, 2.0, "world")],
        )

    def test_plain_format_does_not_request_timestamps(self):
        root, model = _whisper_root()
        pipe = _FakeWhisperPipe()
        be = _DispatchBackend(root, pipe)
        be.transcribe(model, _wav_bytes(), response_format="text")
        self.assertNotIn("return_timestamps", pipe.calls[0]["kwargs"])

    def test_non_whisper_model_is_unsupported_without_pipeline(self):
        root, model = _llm_root()
        be = _DispatchBackend(root, _FakeWhisperPipe())
        with self.assertRaises(UnsupportedModel):
            be.transcribe(model, _wav_bytes())
        self.assertEqual(be.pipeline_requests, [])

    def test_garbage_audio_is_invalid_audio_without_pipeline(self):
        root, model = _whisper_root()
        be = _DispatchBackend(root, _FakeWhisperPipe())
        with self.assertRaises(InvalidAudio):
            be.transcribe(model, b"not audio")
        self.assertEqual(be.pipeline_requests, [])

    def test_unknown_model_404s_before_anything(self):
        root, _ = _whisper_root()
        be = _DispatchBackend(root, _FakeWhisperPipe())
        with self.assertRaises(ModelNotFound):
            be.transcribe("org/ghost", _wav_bytes())
        self.assertEqual(be.pipeline_requests, [])


if __name__ == "__main__":
    unittest.main()
