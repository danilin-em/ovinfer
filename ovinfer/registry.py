"""Model registry: scan models_root (org/model layout) and resolve model ids."""

import json
from pathlib import Path

from .errors import ModelNotFound

# Sentence-transformers 1_Pooling/config.json booleans, in priority order,
# mapped to og pooling tokens. Exactly one is normally true; if several are,
# last-token wins, then cls, then mean.
_ST_POOLING_FLAGS = (
    ("pooling_mode_lasttoken", "last_token"),
    ("pooling_mode_cls_token", "cls"),
    ("pooling_mode_mean_tokens", "mean"),
)


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

    Returns False (not an error) for a nonexistent path.
    """
    return (model_dir / "openvino_vision_embeddings_model.xml").is_file()


def _pooling_from_st_config(model_dir: Path) -> str | None:
    """Read sentence-transformers 1_Pooling/config.json, if present.

    Returns an og pooling token ("cls"/"mean"/"last_token") for the first
    recognized flag set true, or None if the file is absent, unparseable, or
    declares no recognized mode (so the caller falls back to the architecture).
    """
    path = model_dir / "1_Pooling" / "config.json"
    try:
        data = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    for flag, token in _ST_POOLING_FLAGS:
        if data.get(flag):
            return token
    return None


def embedding_pooling(model_dir: Path) -> str:
    """Pick the pooling for an embedding model: "cls", "mean", or "last_token".

    og.TextEmbeddingPipeline defaults to CLS, which is only correct for encoder
    models (BERT/RoBERTa) that prepend a real [CLS] token. For a causal decoder
    LM (Qwen3-Embedding, gte-Qwen, e5-mistral...) the first token attends only
    to itself, so CLS pooling collapses unrelated texts to near-identical
    vectors (cosine ~1); those models pool the last token instead.

    Order of authority: sentence-transformers 1_Pooling/config.json when present
    (the model's own declaration), else the architecture in config.json
    (*ForCausalLM -> last_token), else CLS (og's default; plain encoders and
    unknown exports). Returns a plain string so this module stays og-free.
    """
    from_st = _pooling_from_st_config(model_dir)
    if from_st is not None:
        return from_st
    try:
        config = json.loads((model_dir / "config.json").read_text())
        architectures = config.get("architectures") or []
    except (OSError, ValueError):
        architectures = []
    if any(str(arch).endswith("ForCausalLM") for arch in architectures):
        return "last_token"
    return "cls"


def is_diffusion_model(model_dir: Path) -> bool:
    """True if the model dir holds a diffusers pipeline export.

    Diffusion exports (Text2Image/Image2Image/Inpainting) ship a
    model_index.json at the model root listing the sub-models (transformer/unet,
    vae, text_encoder, scheduler). Plain LLM and VLM exports do not. Image
    generation requests (POST /v1/images/generations) require this marker;
    a model without it is rejected with 400 rather than failing pipeline load.

    Returns False (not an error) for a nonexistent path.
    """
    return (model_dir / "model_index.json").is_file()


def is_whisper_model(model_dir: Path) -> bool:
    """True if the model dir holds a Whisper ASR export (encoder-decoder).

    Whisper exports ship openvino_encoder_model.xml (a separate audio encoder)
    next to the decoder; plain LLM (openvino_model.xml) and VLM
    (openvino_vision_embeddings_model.xml) exports do not. Audio endpoints
    require this marker; a model without it is rejected with 400 rather than
    failing pipeline load.

    Returns False (not an error) for a nonexistent path.
    """
    return (model_dir / "openvino_encoder_model.xml").is_file()
