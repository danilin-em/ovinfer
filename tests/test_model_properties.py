"""Tests for per-model construction properties: config: -> pipeline kwargs.

Backend._properties_for is pure (og-free). _create_pipeline is exercised by
injecting a fake `openvino_genai` module into sys.modules, so no OpenVINO env
is needed; the fake pipeline classes record their positional args and kwargs.
"""

import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock

from ovinfer.backend import Backend


def _root_with(*model_ids: str) -> Path:
    root = Path(tempfile.mkdtemp())
    for mid in model_ids:
        d = root / mid
        d.mkdir(parents=True)
        (d / "openvino_model.xml").write_text("<net/>")
    return root


def _backend(root: Path, cfg_text: str) -> Backend:
    path = root / "ovinfer.yml"
    path.write_text(cfg_text)
    with mock.patch.dict(os.environ, {}, clear=True):
        return Backend(models_root=root, config_file=path)


class _RecordingPipe:
    """Base fake pipeline: records constructor args/kwargs on the subclass."""

    instances: list = []

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs
        type(self).instances.append(self)


def _fake_og() -> types.ModuleType:
    """A minimal fake openvino_genai exposing the two constructor shapes used."""
    og = types.ModuleType("openvino_genai")

    class LLMPipeline(_RecordingPipe):
        instances: list = []

    class _PoolingType:
        CLS = "CLS"
        MEAN = "MEAN"
        LAST_TOKEN = "LAST_TOKEN"

    class _EmbConfig:
        def __init__(self):
            self.pooling_type = None

    class TextEmbeddingPipeline(_RecordingPipe):
        instances: list = []
        Config = _EmbConfig
        PoolingType = _PoolingType

    og.LLMPipeline = LLMPipeline
    og.TextEmbeddingPipeline = TextEmbeddingPipeline
    return og


class PropertiesForTest(unittest.TestCase):
    def test_returns_configured_properties(self):
        root = _root_with("org/a")
        be = _backend(
            root,
            "autoload: false\nmodels:\n  org/a:\n    kind: llm\n    config:\n"
            "      PERFORMANCE_HINT: LATENCY\n      NUM_STREAMS: 2\n",
        )
        self.assertEqual(
            be._properties_for("org/a"),
            {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": 2},
        )

    def test_unconfigured_model_is_empty(self):
        root = _root_with("org/a")
        be = _backend(root, "autoload: true\nmodels:\n  org/a:\n    kind: llm\n")
        self.assertEqual(be._properties_for("org/other"), {})

    def test_model_without_config_is_empty(self):
        root = _root_with("org/a")
        be = _backend(root, "autoload: false\nmodels:\n  org/a:\n    kind: llm\n")
        self.assertEqual(be._properties_for("org/a"), {})


class CreatePipelineForwardingTest(unittest.TestCase):
    def test_llm_pipeline_receives_properties_as_kwargs(self):
        root = _root_with("org/a")
        be = _backend(
            root,
            "autoload: false\nmodels:\n  org/a:\n    kind: llm\n    config:\n"
            "      PERFORMANCE_HINT: LATENCY\n      NUM_STREAMS: 2\n",
        )
        og = _fake_og()
        with mock.patch.dict(sys.modules, {"openvino_genai": og}):
            be._create_pipeline(root / "org/a", "org/a", "llm")
        inst = og.LLMPipeline.instances[-1]
        self.assertEqual(inst.args, (root / "org/a", "CPU"))
        self.assertEqual(inst.kwargs, {"PERFORMANCE_HINT": "LATENCY", "NUM_STREAMS": 2})

    def test_llm_pipeline_without_config_gets_no_kwargs(self):
        root = _root_with("org/a")
        be = _backend(root, "autoload: false\nmodels:\n  org/a:\n    kind: llm\n")
        og = _fake_og()
        with mock.patch.dict(sys.modules, {"openvino_genai": og}):
            be._create_pipeline(root / "org/a", "org/a", "llm")
        self.assertEqual(og.LLMPipeline.instances[-1].kwargs, {})

    def test_embedding_pipeline_receives_properties_after_config(self):
        root = _root_with("org/a")
        be = _backend(
            root,
            "autoload: false\nmodels:\n  org/a:\n    kind: embedding\n    config:\n"
            "      CACHE_DIR: /tmp/cache\n",
        )
        og = _fake_og()
        with mock.patch.dict(sys.modules, {"openvino_genai": og}):
            be._create_pipeline(root / "org/a", "org/a", "embedding")
        inst = og.TextEmbeddingPipeline.instances[-1]
        self.assertEqual(inst.args[:2], (root / "org/a", "CPU"))
        self.assertIsInstance(inst.args[2], og.TextEmbeddingPipeline.Config)
        self.assertEqual(inst.kwargs, {"CACHE_DIR": "/tmp/cache"})


if __name__ == "__main__":
    unittest.main()
