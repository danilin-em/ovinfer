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
