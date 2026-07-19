"""Tests for YAML config file parsing and path resolution.

og is never imported. load_config_file is pure file introspection over
temp YAML files, so it is tested without any OpenVINO environment.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ovinfer import config
from ovinfer.errors import ConfigError


def _write(text: str) -> Path:
    root = Path(tempfile.mkdtemp())
    path = root / "ovinfer.yml"
    path.write_text(text)
    return path


class LoadConfigFileTest(unittest.TestCase):
    def test_full_file_parses_into_internal_types(self):
        path = _write(
            "models_root: /data/models\n"
            "default_device: GPU\n"
            "max_pipelines: 4\n"
            "use_continuous_batching: true\n"
            "device_map:\n"
            "  org/a: NPU\n"
            "  org/b: GPU\n"
            "preload:\n"
            "  org/a: llm\n"
            "  org/b: embedding\n"
        )
        cfg = config.load_config_file(path)
        self.assertEqual(cfg["models_root"], "/data/models")
        self.assertEqual(cfg["default_device"], "GPU")
        self.assertEqual(cfg["max_pipelines"], 4)
        self.assertIs(cfg["use_continuous_batching"], True)
        self.assertEqual(cfg["device_map"], {"org/a": "NPU", "org/b": "GPU"})
        self.assertEqual(cfg["preload"], [("org/a", "llm"), ("org/b", "embedding")])

    def test_preload_preserves_file_order(self):
        path = _write("preload:\n  org/z: llm\n  org/a: embedding\n")
        cfg = config.load_config_file(path)
        self.assertEqual(cfg["preload"], [("org/z", "llm"), ("org/a", "embedding")])

    def test_empty_file_is_empty_dict(self):
        self.assertEqual(config.load_config_file(_write("")), {})

    def test_missing_file_raises(self):
        with self.assertRaises(ConfigError):
            config.load_config_file(Path(tempfile.mkdtemp()) / "nope.yml")

    def test_non_mapping_top_level_raises(self):
        with self.assertRaises(ConfigError):
            config.load_config_file(_write("- a\n- b\n"))

    def test_unknown_key_raises(self):
        with self.assertRaises(ConfigError):
            config.load_config_file(_write("listen: 0.0.0.0:8000\n"))

    def test_invalid_preload_kind_raises(self):
        with self.assertRaises(ConfigError):
            config.load_config_file(_write("preload:\n  org/a: bogus\n"))

    def test_null_preload_kind_raises(self):
        with self.assertRaises(ConfigError):
            config.load_config_file(_write("preload:\n  org/a:\n"))

    def test_empty_device_raises(self):
        with self.assertRaises(ConfigError):
            config.load_config_file(_write("device_map:\n  org/a:\n"))

    def test_bool_max_pipelines_raises(self):
        # bool is a subclass of int in Python; it must not pass as max_pipelines.
        with self.assertRaises(ConfigError):
            config.load_config_file(_write("max_pipelines: true\n"))

    def test_non_bool_continuous_batching_raises(self):
        with self.assertRaises(ConfigError):
            config.load_config_file(_write("use_continuous_batching: 1\n"))

    def test_null_models_root_raises(self):
        with self.assertRaises(ConfigError):
            config.load_config_file(_write("models_root:\n"))

    def test_null_default_device_raises(self):
        with self.assertRaises(ConfigError):
            config.load_config_file(_write("default_device:\n"))


class ResolveConfigPathTest(unittest.TestCase):
    def test_cli_beats_env(self):
        with mock.patch.dict(os.environ, {"OVINFER_CONFIG": "/env.yml"}):
            self.assertEqual(
                config.resolve_config_path("/cli.yml"), Path("/cli.yml")
            )

    def test_env_used_when_no_cli(self):
        with mock.patch.dict(os.environ, {"OVINFER_CONFIG": "/env.yml"}):
            self.assertEqual(config.resolve_config_path(None), Path("/env.yml"))

    def test_none_when_neither(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(config.resolve_config_path(None))


class LoadSettingsPrecedenceTest(unittest.TestCase):
    def _clean_env(self):
        return mock.patch.dict(os.environ, {}, clear=True)

    def test_file_value_used_when_no_cli_no_env(self):
        path = _write("default_device: GPU\n")
        with self._clean_env():
            s = config.load_settings(config_file=path)
        self.assertEqual(s.default_device, "GPU")

    def test_env_overrides_file(self):
        path = _write("default_device: GPU\n")
        with mock.patch.dict(os.environ, {"OVINFER_DEVICE": "NPU"}, clear=True):
            s = config.load_settings(config_file=path)
        self.assertEqual(s.default_device, "NPU")

    def test_cli_overrides_env_and_file(self):
        path = _write("default_device: GPU\n")
        with mock.patch.dict(os.environ, {"OVINFER_DEVICE": "NPU"}, clear=True):
            s = config.load_settings(default_device="CPU", config_file=path)
        self.assertEqual(s.default_device, "CPU")

    def test_default_used_when_absent_everywhere(self):
        path = _write("default_device: GPU\n")  # unrelated key set
        with self._clean_env():
            s = config.load_settings(config_file=path)
        self.assertEqual(s.models_root, config.default_models_root())

    def test_file_device_map_and_preload_reach_settings(self):
        path = _write(
            "device_map:\n  org/a: NPU\n"
            "preload:\n  org/a: embedding\n"
            "max_pipelines: 3\n"
        )
        with self._clean_env():
            s = config.load_settings(config_file=path)
        self.assertEqual(s.device_map, {"org/a": "NPU"})
        self.assertEqual(s.preload, [("org/a", "embedding")])
        self.assertEqual(s.max_pipelines, 3)

    def test_env_device_map_overrides_file_device_map(self):
        path = _write("device_map:\n  org/a: NPU\n")
        with mock.patch.dict(
            os.environ, {"OVINFER_DEVICE_MAP": "org/b=GPU"}, clear=True
        ):
            s = config.load_settings(config_file=path)
        self.assertEqual(s.device_map, {"org/b": "GPU"})

    def test_no_config_file_matches_current_behavior(self):
        with self._clean_env():
            s = config.load_settings()
        self.assertEqual(s.default_device, "CPU")
        self.assertEqual(s.device_map, {})
        self.assertEqual(s.preload, [])


if __name__ == "__main__":
    unittest.main()
