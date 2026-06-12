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


if __name__ == "__main__":
    unittest.main()
