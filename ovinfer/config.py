"""Settings resolution: explicit arguments > OVINFER_* env > defaults.

The OVINFER_ prefix is deliberate: the short OV_ prefix is taken by
OpenVINO's own environment variables.
"""

import os
from dataclasses import dataclass
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
        "device_map",
        "preload",
        "max_pipelines",
        "use_continuous_batching",
    )
)


def resolve_config_path(cli_path: str | Path | None) -> Path | None:
    """Config file path: --config CLI value > OVINFER_CONFIG env > None."""
    if cli_path is not None:
        return Path(cli_path).expanduser()
    env_path = os.environ.get(_ENV_PREFIX + "CONFIG")
    return Path(env_path).expanduser() if env_path else None


def _file_device_map(raw: object) -> dict[str, str]:
    if not isinstance(raw, dict):
        raise ConfigError(
            "device_map must be a mapping model_id: device, "
            f"got {type(raw).__name__}"
        )
    out: dict[str, str] = {}
    for model_id, device in raw.items():
        model_id = str(model_id).strip()
        device = str(device).strip() if device is not None else ""
        if not model_id or not device:
            raise ConfigError(
                f"invalid device_map entry {model_id!r}: {device!r} "
                "(model and device required)"
            )
        out[model_id] = device
    return out


def _file_preload(raw: object) -> list[tuple[str, str]]:
    if not isinstance(raw, dict):
        raise ConfigError(
            f"preload must be a mapping model_id: kind, got {type(raw).__name__}"
        )
    out: list[tuple[str, str]] = []
    for model_id, kind in raw.items():
        model_id = str(model_id).strip()
        kind = str(kind).strip() if kind is not None else ""
        if not model_id or kind not in PIPELINE_KINDS:
            raise ConfigError(
                f"invalid preload entry {model_id!r}: {kind!r} "
                f"(kind one of {PIPELINE_KINDS})"
            )
        out.append((model_id, kind))
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
        if raw["models_root"] is None:
            raise ConfigError("models_root must not be empty")
        cfg["models_root"] = str(raw["models_root"])
    if "default_device" in raw:
        if raw["default_device"] is None:
            raise ConfigError("default_device must not be empty")
        cfg["default_device"] = str(raw["default_device"])
    if "device_map" in raw:
        cfg["device_map"] = _file_device_map(raw["device_map"])
    if "preload" in raw:
        cfg["preload"] = _file_preload(raw["preload"])
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
    return cfg


@dataclass(frozen=True)
class Settings:
    models_root: Path
    default_device: str
    device_map: dict[str, str]
    preload: list[tuple[str, str]]
    max_pipelines: int
    use_continuous_batching: bool


def load_settings(
    *,
    models_root: Path | str | None = None,
    default_device: str | None = None,
    device_map: dict[str, str] | None = None,
    preload: list[tuple[str, str]] | None = None,
    max_pipelines: int | None = None,
    use_continuous_batching: bool | None = None,
) -> Settings:
    """Resolve effective settings; raises ConfigError on invalid combinations."""
    env = os.environ
    if models_root is None:
        models_root = env.get(_ENV_PREFIX + "MODELS_ROOT") or default_models_root()
    if default_device is None:
        default_device = env.get(_ENV_PREFIX + "DEVICE") or "CPU"
    if device_map is None:
        device_map = parse_device_map(env.get(_ENV_PREFIX + "DEVICE_MAP", ""))
    if preload is None:
        preload = parse_preload(env.get(_ENV_PREFIX + "PRELOAD", ""))
    if use_continuous_batching is None:
        use_continuous_batching = (
            env.get(_ENV_PREFIX + "USE_CONTINUOUS_BATCHING", "0") == "1"
        )
    if max_pipelines is None:
        raw_max = env.get(_ENV_PREFIX + "MAX_PIPELINES")
        if raw_max is None:
            max_pipelines = max(2, len(preload))
        else:
            try:
                max_pipelines = int(raw_max)
            except ValueError:
                raise ConfigError(
                    f"OVINFER_MAX_PIPELINES must be an integer, got {raw_max!r}"
                ) from None
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
        device_map=dict(device_map),
        preload=list(preload),
        max_pipelines=max_pipelines,
        use_continuous_batching=use_continuous_batching,
    )
