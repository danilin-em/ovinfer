"""Settings resolution: explicit arguments > OVINFER_* env > defaults.

The OVINFER_ prefix is deliberate: the short OV_ prefix is taken by
OpenVINO's own environment variables.
"""

import os
from dataclasses import dataclass, field
from pathlib import Path

import yaml

from .errors import ConfigError

_ENV_PREFIX = "OVINFER_"
PIPELINE_KINDS = (
    "llm",
    "vlm",
    "embedding",
    "rerank",
    "text2image",
    "image2image",
    "inpainting",
    "whisper",
    "tts",
)


@dataclass(frozen=True)
class ModelConfig:
    """One per-model config entry from the file's `models:` map."""

    kind: str                       # required; in PIPELINE_KINDS
    device: str | None = None       # None -> default_device
    preload: bool = False           # warm at startup
    config: dict = field(default_factory=dict)   # reserved stub, ignored


_MODEL_ENTRY_KEYS = frozenset(("kind", "device", "preload", "config"))


def _xdg_data_home() -> Path:
    raw = os.environ.get("XDG_DATA_HOME", "")
    return Path(raw) if raw else Path.home() / ".local" / "share"


def default_models_root() -> Path:
    return _xdg_data_home() / "ovinfer" / "models"


def parse_device_map(raw: str) -> dict[str, str]:
    """Parse 'org/model=NPU,org/model2=GPU' into {model_id: device}."""
    device_map: dict[str, str] = {}
    for entry in filter(None, (e.strip() for e in raw.split(","))):
        model_id, sep, device = entry.partition("=")
        if not sep or not model_id.strip() or not device.strip():
            raise ConfigError(
                f"Invalid device-map entry {entry!r} (expected MODEL=DEVICE)"
            )
        device_map[model_id.strip()] = device.strip()
    return device_map


def parse_preload(raw: str) -> list[tuple[str, str]]:
    """Parse 'org/model[:kind],...' into (model_id, kind); kind defaults to 'llm'."""
    entries: list[tuple[str, str]] = []
    for item in filter(None, (e.strip() for e in raw.split(","))):
        model_id, sep, kind = item.partition(":")
        kind = kind.strip() if sep else "llm"
        if not model_id.strip() or kind not in PIPELINE_KINDS:
            raise ConfigError(
                f"Invalid preload entry {item!r} "
                f"(expected MODEL[:kind], kind one of {PIPELINE_KINDS})"
            )
        entries.append((model_id.strip(), kind))
    return entries


_FILE_KEYS = frozenset(
    (
        "models_root",
        "default_device",
        "max_pipelines",
        "use_continuous_batching",
        "autoload",
        "models",
    )
)


def resolve_config_path(cli_path: str | Path | None) -> Path | None:
    """Config file path: --config CLI value > OVINFER_CONFIG env > None."""
    if cli_path is not None:
        return Path(cli_path).expanduser()
    env_path = os.environ.get(_ENV_PREFIX + "CONFIG")
    return Path(env_path).expanduser() if env_path else None


def _file_models(raw: object) -> dict[str, "ModelConfig"]:
    """Parse and validate the file's `models:` map into ModelConfig objects."""
    if not isinstance(raw, dict):
        raise ConfigError(
            f"models must be a mapping model_id: entry, got {type(raw).__name__}"
        )
    out: dict[str, ModelConfig] = {}
    for model_id, entry in raw.items():
        model_id = str(model_id).strip()
        if not model_id:
            raise ConfigError("models has an entry with an empty model id")
        if not isinstance(entry, dict):
            raise ConfigError(
                f"models entry {model_id!r} must be a mapping, "
                f"got {type(entry).__name__}"
            )
        unknown = set(entry) - _MODEL_ENTRY_KEYS
        if unknown:
            raise ConfigError(
                f"models entry {model_id!r} has unknown keys: {sorted(unknown)}"
            )
        kind = entry.get("kind")
        kind = str(kind).strip() if kind is not None else ""
        if kind not in PIPELINE_KINDS:
            raise ConfigError(
                f"models entry {model_id!r}: kind is required and must be one of "
                f"{PIPELINE_KINDS}, got {entry.get('kind')!r}"
            )
        device = entry.get("device")
        if "device" in entry:
            if device is None or isinstance(device, (dict, list)) or not str(device).strip():
                raise ConfigError(
                    f"models entry {model_id!r}: device must be a non-empty scalar"
                )
            device = str(device).strip()
        preload = entry.get("preload", False)
        if not isinstance(preload, bool):
            raise ConfigError(
                f"models entry {model_id!r}: preload must be a boolean, got {preload!r}"
            )
        cfg = entry.get("config", {})
        if not isinstance(cfg, dict):
            raise ConfigError(
                f"models entry {model_id!r}: config must be a mapping, "
                f"got {type(cfg).__name__}"
            )
        out[model_id] = ModelConfig(
            kind=kind, device=device, preload=preload, config=dict(cfg)
        )
    return out


def load_config_file(path: str | Path) -> dict:
    """Parse and validate a YAML config file into internal setting types.

    Returns a dict whose keys are a subset of _FILE_KEYS. Raises ConfigError
    on a missing file, malformed YAML, an unknown key, or an invalid value.
    """
    p = Path(path).expanduser()
    if not p.is_file():
        raise ConfigError(f"config file not found: {p}")
    try:
        raw = yaml.safe_load(p.read_text())
    except yaml.YAMLError as exc:
        raise ConfigError(f"invalid YAML in {p}: {exc}") from None
    if raw is None:
        return {}
    if not isinstance(raw, dict):
        raise ConfigError(
            f"config file {p} must be a mapping at top level, "
            f"got {type(raw).__name__}"
        )
    unknown = set(raw) - _FILE_KEYS
    if unknown:
        raise ConfigError(f"unknown config keys: {sorted(unknown)}")
    cfg: dict = {}
    if "models_root" in raw:
        value = raw["models_root"]
        if value is None or isinstance(value, (dict, list)) or not str(value).strip():
            raise ConfigError("models_root must be a non-empty scalar")
        cfg["models_root"] = str(value).strip()
    if "default_device" in raw:
        value = raw["default_device"]
        if value is None or isinstance(value, (dict, list)) or not str(value).strip():
            raise ConfigError("default_device must be a non-empty scalar")
        cfg["default_device"] = str(value).strip()
    if "max_pipelines" in raw:
        mp = raw["max_pipelines"]
        if isinstance(mp, bool) or not isinstance(mp, int):
            raise ConfigError(f"max_pipelines must be an integer, got {mp!r}")
        cfg["max_pipelines"] = mp
    if "use_continuous_batching" in raw:
        cb = raw["use_continuous_batching"]
        if not isinstance(cb, bool):
            raise ConfigError(
                f"use_continuous_batching must be a boolean, got {cb!r}"
            )
        cfg["use_continuous_batching"] = cb
    if "autoload" in raw:
        al = raw["autoload"]
        if not isinstance(al, bool):
            raise ConfigError(f"autoload must be a boolean, got {al!r}")
        cfg["autoload"] = al
    if "models" in raw:
        cfg["models"] = _file_models(raw["models"])
    return cfg


@dataclass(frozen=True)
class Settings:
    models_root: Path
    default_device: str
    max_pipelines: int
    use_continuous_batching: bool
    autoload: bool
    models: dict[str, ModelConfig]
    device_map: dict[str, str]          # derived from models
    preload: list[tuple[str, str]]      # derived from models


def load_settings(
    *,
    models_root: Path | str | None = None,
    default_device: str | None = None,
    max_pipelines: int | None = None,
    use_continuous_batching: bool | None = None,
    autoload: bool | None = None,
    config_file: str | Path | None = None,
) -> Settings:
    """Resolve effective settings; raises ConfigError on invalid combinations.

    Per scalar field: explicit argument (CLI) > OVINFER_* env > YAML config file
    > built-in default. The `models:` map is file-only; `device_map` and
    `preload` are derived from it. `autoload` defaults to "no config file given".
    """
    env = os.environ
    config_path = resolve_config_path(config_file)
    file_cfg = load_config_file(config_path) if config_path is not None else {}

    if models_root is None:
        models_root = (
            env.get(_ENV_PREFIX + "MODELS_ROOT")
            or file_cfg.get("models_root")
            or default_models_root()
        )
    if default_device is None:
        default_device = (
            env.get(_ENV_PREFIX + "DEVICE")
            or file_cfg.get("default_device")
            or "CPU"
        )
    if use_continuous_batching is None:
        env_cb = env.get(_ENV_PREFIX + "USE_CONTINUOUS_BATCHING")
        if env_cb is not None:
            use_continuous_batching = env_cb == "1"
        else:
            use_continuous_batching = file_cfg.get("use_continuous_batching", False)
    if autoload is None:
        env_al = env.get(_ENV_PREFIX + "AUTOLOAD")
        if env_al is not None:
            autoload = env_al == "1"
        elif "autoload" in file_cfg:
            autoload = file_cfg["autoload"]
        else:
            autoload = config_path is None

    models: dict[str, ModelConfig] = file_cfg.get("models", {})
    device_map = {mid: mc.device for mid, mc in models.items() if mc.device}
    preload = [(mid, mc.kind) for mid, mc in models.items() if mc.preload]

    if max_pipelines is None:
        raw_max = env.get(_ENV_PREFIX + "MAX_PIPELINES")
        if raw_max is not None:
            try:
                max_pipelines = int(raw_max)
            except ValueError:
                raise ConfigError(
                    f"OVINFER_MAX_PIPELINES must be an integer, got {raw_max!r}"
                ) from None
        elif "max_pipelines" in file_cfg:
            max_pipelines = file_cfg["max_pipelines"]
        else:
            max_pipelines = max(2, len(preload))
    if max_pipelines < 1:
        raise ConfigError(f"max_pipelines must be >= 1, got {max_pipelines}")
    if len(preload) > max_pipelines:
        raise ConfigError(
            f"{len(preload)} preload models exceed max_pipelines={max_pipelines}; "
            "raise the limit or trim the preload list"
        )
    return Settings(
        models_root=Path(models_root).expanduser(),
        default_device=default_device,
        max_pipelines=max_pipelines,
        use_continuous_batching=use_continuous_batching,
        autoload=autoload,
        models=dict(models),
        device_map=device_map,
        preload=preload,
    )
