"""Tests for the autoload whitelist gate (og-free; no _pipeline is entered)."""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ovinfer.backend import Backend
from ovinfer.errors import ModelNotFound


def _root_with(*model_ids: str) -> Path:
    root = Path(tempfile.mkdtemp())
    for mid in model_ids:
        d = root / mid
        d.mkdir(parents=True)
        (d / "openvino_model.xml").write_text("<net/>")
    return root


def _cfg(root: Path, text: str) -> Path:
    path = root / "ovinfer.yml"
    path.write_text(text)
    return path


def _backend(root: Path, cfg_text: str) -> Backend:
    cfg = _cfg(root, cfg_text)
    with mock.patch.dict(os.environ, {}, clear=True):
        return Backend(models_root=root, config_file=cfg)


class AutoloadGateTest(unittest.TestCase):
    def test_whitelist_resolves_configured_model(self):
        root = _root_with("org/a", "org/b")
        be = _backend(root, "autoload: false\nmodels:\n  org/a:\n    kind: llm\n")
        self.assertEqual(be._resolve("org/a"), root / "org/a")

    def test_whitelist_rejects_unconfigured_model_on_disk(self):
        root = _root_with("org/a", "org/b")
        be = _backend(root, "autoload: false\nmodels:\n  org/a:\n    kind: llm\n")
        with self.assertRaises(ModelNotFound):
            be._resolve("org/b")

    def test_whitelist_gate_flows_through_endpoint_methods(self):
        root = _root_with("org/a", "org/b")
        be = _backend(root, "autoload: false\nmodels:\n  org/a:\n    kind: rerank\n")
        with self.assertRaises(ModelNotFound):
            be.rerank("org/b", "q", ["d"])

    def test_list_models_returns_configured_ids_when_gated(self):
        root = _root_with("org/a", "org/b")
        be = _backend(root, "autoload: false\nmodels:\n  org/a:\n    kind: llm\n")
        self.assertEqual(be.list_models(), ["org/a"])

    def test_autoload_true_resolves_unconfigured_model(self):
        root = _root_with("org/a", "org/b")
        be = _backend(root, "autoload: true\nmodels:\n  org/a:\n    kind: llm\n")
        self.assertEqual(be._resolve("org/b"), root / "org/b")

    def test_autoload_true_list_models_is_disk_scan(self):
        root = _root_with("org/a", "org/b")
        be = _backend(root, "autoload: true\nmodels:\n  org/a:\n    kind: llm\n")
        self.assertEqual(be.list_models(), ["org/a", "org/b"])

    def test_validate_raises_on_missing_configured_dir(self):
        root = _root_with("org/a")
        be = _backend(
            root,
            "autoload: false\nmodels:\n"
            "  org/a:\n    kind: llm\n  org/ghost:\n    kind: llm\n",
        )
        with self.assertRaises(ModelNotFound):
            be.validate_configured_models()

    def test_validate_passes_when_all_present(self):
        root = _root_with("org/a", "org/b")
        be = _backend(
            root,
            "autoload: false\nmodels:\n"
            "  org/a:\n    kind: llm\n  org/b:\n    kind: embedding\n",
        )
        be.validate_configured_models()  # no raise


if __name__ == "__main__":
    unittest.main()
