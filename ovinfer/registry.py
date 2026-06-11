"""Model registry: scan models_root (org/model layout) and resolve model ids."""

from pathlib import Path

from .errors import ModelNotFound


def list_models(models_root: Path) -> list[str]:
    """Scan models_root two levels deep (org/model layout).

    A directory counts as a model if it contains at least one OpenVINO IR
    (*.xml) anywhere inside (diffusion models keep IRs in subdirectories).
    """
    models: list[str] = []
    if not models_root.is_dir():
        return models
    for org in sorted(models_root.iterdir()):
        if not org.is_dir():
            continue
        for model in sorted(org.iterdir()):
            if model.is_dir() and next(model.rglob("*.xml"), None) is not None:
                models.append(f"{org.name}/{model.name}")
    return models


def resolve_model_dir(models_root: Path, model_id: str) -> Path:
    """Map the OpenAI `model` field to a local directory or raise ModelNotFound.

    Membership check against list_models() also blocks path traversal.
    """
    if model_id not in list_models(models_root):
        raise ModelNotFound(model_id, models_root)
    return models_root / model_id


def model_created(models_root: Path, model_id: str) -> int:
    """Best-effort mtime; 0 if the dir vanished between scan and stat."""
    try:
        return int((models_root / model_id).stat().st_mtime)
    except OSError:
        return 0


def is_vlm_model(model_dir: Path) -> bool:
    """True if the model dir holds a VLM export (a vision-encoder IR).

    VLM exports ship openvino_vision_embeddings_model.xml alongside the
    language model; plain LLM exports do not. Chat requests for VLM models go
    to og.VLMPipeline (which also serves text-only prompts) rather than
    og.LLMPipeline, whose openvino_model.xml a VLM export lacks.
    """
    return (model_dir / "openvino_vision_embeddings_model.xml").is_file()
