"""Conservative NPU-incompatibility preflight.

Predicts, from a model's on-disk metadata, whether placing it on NPU will fail
— so the server can warn clearly up front instead of surfacing a raw NPU
compiler traceback (group-wise int4 LLMs) or a request-time 500 (embeddings).

Warn-only by contract: the caller still attempts NPU. The heuristic only flags
failures verified on this stack; anything uncertain returns None and is left to
the compiler to decide.

No openvino import: rt_info is read with a light regex over openvino_model.xml,
keeping this module cheap to import (mirrors backend's lazy openvino_genai import).
"""

import logging
import re
from pathlib import Path

log = logging.getLogger("ovinfer")

# Anchored on the leading '<' so it never matches <backup_mode value="...">,
# which nncf writes right next to the real <mode>.
_MODE_RE = re.compile(r'<mode value="([^"]+)"')
_GROUP_SIZE_RE = re.compile(r'<group_size value="(-?\d+)"')


def _weight_compression(model_dir: Path) -> tuple[str | None, str | None]:
    """(mode, group_size) from the nncf weight_compression rt_info, or (None, None)."""
    try:
        text = (model_dir / "openvino_model.xml").read_text(
            encoding="utf-8", errors="ignore"
        )
    except OSError:
        return None, None
    mode = _MODE_RE.search(text)
    group_size = _GROUP_SIZE_RE.search(text)
    return (mode.group(1) if mode else None, group_size.group(1) if group_size else None)


def reject_reason(model_dir: Path, kind: str) -> str | None:
    """Why this (model, kind) will fail on NPU, or None if it should be left alone.

    Conservative: only flags empirically verified failures. Callers apply this
    only once the resolved device is NPU.
    """
    if kind in ("embedding", "rerank"):
        return (
            "embedding and rerank pipelines need dynamic input shapes, but NPU "
            "requires static shapes (NPUW) — inference fails at request time"
        )
    if kind == "llm":
        mode, group_size = _weight_compression(model_dir)
        if mode and mode.startswith("int4") and group_size not in (None, "-1"):
            return (
                f"int4 weights are group-wise (group_size={group_size}); NPU "
                "needs channel-wise int4 (group_size=-1) — compilation fails"
            )
    return None


def warn_if_incompatible(
    model_id: str, model_dir: Path, kind: str, device: str
) -> None:
    """Log one clear WARNING when an NPU placement is known to fail; else silent.

    Warn-only: the caller proceeds to build the pipeline regardless. A no-op for
    non-NPU devices and for placements the heuristic cannot vouch against.
    """
    if not device.upper().startswith("NPU"):
        return
    reason = reject_reason(model_dir, kind)
    if reason:
        log.warning(
            "%s will likely fail on NPU: %s. Attempting anyway.", model_id, reason
        )
