"""Settings resolution: explicit arguments > OVINFER_* env > defaults.

The OVINFER_ prefix is deliberate: the short OV_ prefix is taken by
OpenVINO's own environment variables.
"""

import os
from dataclasses import dataclass
from pathlib import Path

from .errors import ConfigError

_ENV_PREFIX = "OVINFER_"
PIPELINE_KINDS = ("llm", "vlm", "embedding", "rerank", "text2image", "image2image")


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
