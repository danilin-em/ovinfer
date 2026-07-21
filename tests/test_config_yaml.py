"""Tests for YAML config file parsing and settings precedence.

og is never imported. load_config_file / load_settings are pure file
introspection over temp YAML files, so they run without any OpenVINO env.
"""

import os
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from ovinfer import config
from ovinfer.config import ModelConfig
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
            "autoload: true\n"
            "models:\n"
            "  org/a:\n"
            "    kind: llm\n"
            "    device: NPU\n"
            "    preload: true\n"
            "  org/b:\n"
            "    kind: embedding\n"
        )
        cfg = config.load_config_file(path)
        self.assertEqual(cfg["models_root"], "/data/models")
        self.assertEqual(cfg["default_device"], "GPU")
        self.assertEqual(cfg["max_pipelines"], 4)
        self.assertIs(cfg["use_continuous_batching"], True)
        self.assertIs(cfg["autoload"], True)
        self.assertEqual(
            cfg["models"],
            {
                "org/a": ModelConfig(kind="llm", device="NPU", preload=True),
                "org/b": ModelConfig(kind="embedding"),
            },
        )

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

    def test_legacy_device_map_key_is_unknown(self):
        with self.assertRaises(ConfigError):
            config.load_config_file(_write("device_map:\n  org/a: NPU\n"))

    def test_legacy_preload_key_is_unknown(self):
        with self.assertRaises(ConfigError):
            config.load_config_file(_write("preload:\n  org/a: llm\n"))

    def test_models_not_a_mapping_raises(self):
        with self.assertRaises(ConfigError):
            config.load_config_file(_write("models:\n  - org/a\n"))

    def test_model_entry_not_a_mapping_raises(self):
        with self.assertRaises(ConfigError):
            config.load_config_file(_write("models:\n  org/a: llm\n"))

    def test_missing_kind_raises(self):
        with self.assertRaises(ConfigError):
            config.load_config_file(_write("models:\n  org/a:\n    device: NPU\n"))

    def test_invalid_kind_raises(self):
        with self.assertRaises(ConfigError):
            config.load_config_file(_write("models:\n  org/a:\n    kind: bogus\n"))

    def test_unknown_entry_key_raises(self):
        with self.assertRaises(ConfigError):
            config.load_config_file(
                _write("models:\n  org/a:\n    kind: llm\n    gpu: yes\n")
            )

    def test_non_scalar_device_raises(self):
        with self.assertRaises(ConfigError):
            config.load_config_file(
                _write("models:\n  org/a:\n    kind: llm\n    device:\n      x: y\n")
            )

    def test_empty_device_raises(self):
        with self.assertRaises(ConfigError):
            config.load_config_file(
                _write("models:\n  org/a:\n    kind: llm\n    device:\n")
            )

    def test_non_bool_preload_raises(self):
        with self.assertRaises(ConfigError):
            config.load_config_file(
                _write("models:\n  org/a:\n    kind: llm\n    preload: 1\n")
            )

    def test_non_mapping_config_raises(self):
        with self.assertRaises(ConfigError):
            config.load_config_file(
                _write("models:\n  org/a:\n    kind: llm\n    config: nope\n")
            )

    def test_config_stub_is_stored(self):
        cfg = config.load_config_file(
            _write("models:\n  org/a:\n    kind: llm\n    config:\n      k: v\n")
        )
        self.assertEqual(cfg["models"]["org/a"].config, {"k": "v"})

    def test_optional_fields_default(self):
        cfg = config.load_config_file(_write("models:\n  org/a:\n    kind: vlm\n"))
        mc = cfg["models"]["org/a"]
        self.assertIsNone(mc.device)
        self.assertIs(mc.preload, False)
        self.assertEqual(mc.config, {})

    def test_non_bool_autoload_raises(self):
        with self.assertRaises(ConfigError):
            config.load_config_file(_write("autoload: 1\n"))

    def test_bool_max_pipelines_raises(self):
        with self.assertRaises(ConfigError):
            config.load_config_file(_write("max_pipelines: true\n"))

    def test_non_bool_continuous_batching_raises(self):
        with self.assertRaises(ConfigError):
            config.load_config_file(_write("use_continuous_batching: 1\n"))

    def test_null_models_root_raises(self):
        with self.assertRaises(ConfigError):
            config.load_config_file(_write("models_root:\n"))

    def test_empty_string_default_device_raises(self):
        with self.assertRaises(ConfigError):
            config.load_config_file(_write('default_device: ""\n'))

    def test_null_default_device_raises(self):
        with self.assertRaises(ConfigError):
            config.load_config_file(_write("default_device:\n"))

    def test_empty_string_models_root_raises(self):
        with self.assertRaises(ConfigError):
            config.load_config_file(_write('models_root: ""\n'))

    def test_non_scalar_models_root_raises(self):
        with self.assertRaises(ConfigError):
            config.load_config_file(_write("models_root:\n  x: y\n"))


class ResolveConfigPathTest(unittest.TestCase):
    def test_cli_beats_env(self):
        with mock.patch.dict(os.environ, {"OVINFER_CONFIG": "/env.yml"}):
            self.assertEqual(config.resolve_config_path("/cli.yml"), Path("/cli.yml"))

    def test_env_used_when_no_cli(self):
        with mock.patch.dict(os.environ, {"OVINFER_CONFIG": "/env.yml"}):
            self.assertEqual(config.resolve_config_path(None), Path("/env.yml"))

    def test_none_when_neither(self):
        with mock.patch.dict(os.environ, {}, clear=True):
            self.assertIsNone(config.resolve_config_path(None))


class LoadSettingsTest(unittest.TestCase):
    def _clean_env(self):
        return mock.patch.dict(os.environ, {}, clear=True)

    def test_file_value_used_when_no_cli_no_env(self):
        path = _write("default_device: GPU\n")
        with self._clean_env():
            s = config.load_settings(config_file=path)
        self.assertEqual(s.default_device, "GPU")

    def test_cli_overrides_env_and_file(self):
        path = _write("default_device: GPU\n")
        with mock.patch.dict(os.environ, {"OVINFER_DEVICE": "NPU"}, clear=True):
            s = config.load_settings(default_device="CPU", config_file=path)
        self.assertEqual(s.default_device, "CPU")

    def test_models_derive_device_map_and_preload(self):
        path = _write(
            "max_pipelines: 3\n"
            "models:\n"
            "  org/a:\n    kind: llm\n    device: NPU\n    preload: true\n"
            "  org/b:\n    kind: embedding\n"
        )
        with self._clean_env():
            s = config.load_settings(config_file=path)
        self.assertEqual(s.device_map, {"org/a": "NPU"})
        self.assertEqual(s.preload, [("org/a", "llm")])
        self.assertEqual(set(s.models), {"org/a", "org/b"})

    def test_no_config_file_defaults_autoload_true(self):
        with self._clean_env():
            s = config.load_settings()
        self.assertIs(s.autoload, True)
        self.assertEqual(s.device_map, {})
        self.assertEqual(s.preload, [])
        self.assertEqual(s.models, {})

    def test_config_file_defaults_autoload_false(self):
        path = _write("default_device: CPU\n")
        with self._clean_env():
            s = config.load_settings(config_file=path)
        self.assertIs(s.autoload, False)

    def test_file_autoload_key_used(self):
        path = _write("autoload: true\n")
        with self._clean_env():
            s = config.load_settings(config_file=path)
        self.assertIs(s.autoload, True)

    def test_env_autoload_overrides_file(self):
        path = _write("autoload: true\n")
        with mock.patch.dict(os.environ, {"OVINFER_AUTOLOAD": "0"}, clear=True):
            s = config.load_settings(config_file=path)
        self.assertIs(s.autoload, False)

    def test_cli_autoload_overrides_env_and_file(self):
        path = _write("autoload: false\n")
        with mock.patch.dict(os.environ, {"OVINFER_AUTOLOAD": "0"}, clear=True):
            s = config.load_settings(autoload=True, config_file=path)
        self.assertIs(s.autoload, True)

    def test_env_max_pipelines_overrides_file(self):
        path = _write("max_pipelines: 2\n")
        with mock.patch.dict(os.environ, {"OVINFER_MAX_PIPELINES": "7"}, clear=True):
            s = config.load_settings(config_file=path)
        self.assertEqual(s.max_pipelines, 7)

    def test_preload_exceeding_max_pipelines_raises(self):
        path = _write(
            "max_pipelines: 1\n"
            "models:\n"
            "  org/a:\n    kind: llm\n    preload: true\n"
            "  org/b:\n    kind: embedding\n    preload: true\n"
        )
        with self._clean_env():
            with self.assertRaises(ConfigError):
                config.load_settings(config_file=path)

    def test_env_overrides_file(self):
        path = _write("default_device: GPU\n")
        with mock.patch.dict(os.environ, {"OVINFER_DEVICE": "NPU"}, clear=True):
            s = config.load_settings(config_file=path)
        self.assertEqual(s.default_device, "NPU")

    def test_default_used_when_absent_everywhere(self):
        path = _write("default_device: GPU\n")  # unrelated key set
        with self._clean_env():
            s = config.load_settings(config_file=path)
        self.assertEqual(s.models_root, config.default_models_root())

    def test_empty_env_continuous_batching_overrides_file(self):
        path = _write("use_continuous_batching: true\n")
        with mock.patch.dict(
            os.environ, {"OVINFER_USE_CONTINUOUS_BATCHING": ""}, clear=True
        ):
            s = config.load_settings(config_file=path)
        self.assertIs(s.use_continuous_batching, False)


class BackendConfigFileTest(unittest.TestCase):
    def test_backend_reads_config_file(self):
        from ovinfer.backend import Backend

        path = _write("default_device: GPU\nmax_pipelines: 5\n")
        with mock.patch.dict(os.environ, {}, clear=True):
            backend = Backend(config_file=path)
        self.assertEqual(backend.settings.default_device, "GPU")
        self.assertEqual(backend.settings.max_pipelines, 5)
        self.assertIs(backend.settings.autoload, False)


if __name__ == "__main__":
    unittest.main()
