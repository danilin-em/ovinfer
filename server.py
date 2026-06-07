"""OpenAI-compatible HTTP server over openvino-genai pipelines.

Inference is not implemented yet: every backend method validates the request,
resolves the model and raises PipelineNotImplemented (HTTP 501). The HTTP
contract (schemas, routes, errors, streaming) is final.

Pipeline-to-endpoint mapping:
    LLMPipeline                -> POST /v1/chat/completions
    VLMPipeline                -> POST /v1/chat/completions (image_url parts in messages)
    ContinuousBatchingPipeline -> same chat endpoints, alternative backend
                                  (OV_USE_CONTINUOUS_BATCHING=1)
    TextEmbeddingPipeline      -> POST /v1/embeddings
    TextRerankPipeline         -> POST /v1/rerank (Cohere/Jina de-facto standard)
    Text2ImagePipeline         -> POST /v1/images/generations
    Image2ImagePipeline        -> POST /v1/images/edits (image + prompt)
    InpaintingPipeline         -> POST /v1/images/edits (mask present)
    WhisperPipeline            -> POST /v1/audio/transcriptions, /v1/audio/translations
    Text2SpeechPipeline        -> POST /v1/audio/speech
    Text2VideoPipeline         -> POST /v1/videos

Run: openvino-env .venv/bin/python server.py
"""

import json
import os
from pathlib import Path
from typing import Any, Iterator, Literal

from fastapi import FastAPI, File, Form, Request, Response, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict

# --- 1. Config ---------------------------------------------------------------

_DEFAULT_MODELS_ROOT = Path(__file__).resolve().parent / "models"

_raw_port = os.environ.get("OV_SERVER_PORT", "8000")
try:
    OV_SERVER_PORT = int(_raw_port)
except ValueError as exc:
    raise SystemExit(f"OV_SERVER_PORT must be an integer, got: {_raw_port!r}") from exc
OV_MODELS_ROOT = Path(os.environ.get("OV_MODELS_ROOT", str(_DEFAULT_MODELS_ROOT)))
OV_DEVICE = os.environ.get("OV_DEVICE", "CPU")
OV_USE_CONTINUOUS_BATCHING = os.environ.get("OV_USE_CONTINUOUS_BATCHING", "0") == "1"

# --- 1a. Errors (OpenAI error format) -------------------------------------------

class APIError(Exception):
    """Base for errors rendered in the OpenAI error format."""

    status_code = 500
    error_type = "internal_error"
    code = "internal_error"


class ModelNotFound(APIError):
    status_code = 404
    error_type = "invalid_request_error"
    code = "model_not_found"

    def __init__(self, model_id: str) -> None:
        super().__init__(f"Model '{model_id}' not found in {OV_MODELS_ROOT}")


class PipelineNotImplemented(APIError):
    status_code = 501
    error_type = "not_implemented"
    code = "pipeline_not_implemented"

    def __init__(self, pipeline: str) -> None:
        super().__init__(f"{pipeline} backend is not implemented yet")


def error_body(exc: APIError) -> dict:
    return {"error": {"message": str(exc), "type": exc.error_type, "code": exc.code}}


# --- 1b. Model registry ----------------------------------------------------------

def list_models() -> list[str]:
    """Scan OV_MODELS_ROOT two levels deep (org/model layout).

    A directory counts as a model if it contains at least one OpenVINO IR
    (*.xml) anywhere inside (diffusion models keep IRs in subdirectories).
    """
    models: list[str] = []
    if not OV_MODELS_ROOT.is_dir():
        return models
    for org in sorted(OV_MODELS_ROOT.iterdir()):
        if not org.is_dir():
            continue
        for model in sorted(org.iterdir()):
            if model.is_dir() and next(model.rglob("*.xml"), None) is not None:
                models.append(f"{org.name}/{model.name}")
    return models


def resolve_model_dir(model_id: str) -> Path:
    """Map the OpenAI `model` field to a local directory or raise 404.

    Membership check against list_models() also blocks path traversal.

    TODO: cache the list_models() scan when PipelineBackend caching lands.
    """
    if model_id not in list_models():
        raise ModelNotFound(model_id)
    return OV_MODELS_ROOT / model_id


def _model_created(model_id: str) -> int:
    """Best-effort mtime; 0 if the dir vanished between scan and stat."""
    try:
        return int((OV_MODELS_ROOT / model_id).stat().st_mtime)
    except OSError:
        return 0


# --- 2. OpenAI API request schemas ----------------------------------------------
# extra="allow": unknown OpenAI params (logit_bias, seed, user, ...) must not 422.

class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: Literal["system", "user", "assistant", "tool"]
    # Plain string or a list of content parts ({"type": "text"|"image_url", ...});
    # image content parts route the request to VLMPipeline.
    content: str | list[dict[str, Any]] | None = None


class ChatCompletionRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    messages: list[ChatMessage]
    stream: bool = False
    temperature: float | None = None
    top_p: float | None = None
    max_tokens: int | None = None
    max_completion_tokens: int | None = None


class EmbeddingsRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    # OpenAI also accepts pre-tokenized input: token array or batch of arrays.
    input: str | list[str] | list[int] | list[list[int]]


class RerankRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    query: str
    # Cohere/Jina allow plain strings or structured docs ({"text": ...}).
    documents: list[str | dict[str, Any]]
    top_n: int | None = None


class ImagesGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    prompt: str
    n: int = 1
    size: str | None = None
    response_format: Literal["url", "b64_json"] = "b64_json"


class SpeechRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    input: str
    voice: str | None = None
    # Text2SpeechPipeline produces WAV; mp3 (OpenAI's default) would need
    # transcoding, so wav is this server's default.
    response_format: str = "wav"


class VideoGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    prompt: str
    size: str | None = None
    seconds: int | None = None


# --- 3. PipelineBackend -----------------------------------------------------------

_IMAGE_PART_TYPES = {"image_url", "input_image"}


def messages_have_images(messages: list[ChatMessage]) -> bool:
    """True if any message carries an image content part (VLM request)."""
    for msg in messages:
        if isinstance(msg.content, list):
            for part in msg.content:
                if part.get("type") in _IMAGE_PART_TYPES:
                    return True
    return False


class PipelineBackend:
    """All openvino-genai calls live behind this class.

    One stub method per pipeline. Each stub resolves the model first
    (so 404 takes precedence over 501) and raises PipelineNotImplemented
    naming the pipeline that will serve the request. Real pipelines will be
    created lazily on first use and cached in self._cache keyed by
    (model_id, pipeline kind).
    """

    def __init__(self, models_root: Path, device: str) -> None:
        # Use this (not the global OV_MODELS_ROOT) in pipeline constructors.
        self.models_root = models_root
        self.device = device
        self._cache: dict[tuple[str, str], Any] = {}

    # -- chat: LLMPipeline | VLMPipeline | ContinuousBatchingPipeline --

    def chat(self, req: ChatCompletionRequest) -> dict:
        # TODO: model_dir = resolve_model_dir(req.model)
        #       pipe = VLMPipeline(model_dir, self.device) if images present,
        #       else LLMPipeline(model_dir, self.device); when
        #       OV_USE_CONTINUOUS_BATCHING use ContinuousBatchingPipeline.
        #       result = pipe.generate(...); wrap into a chat.completion body.
        resolve_model_dir(req.model)
        raise PipelineNotImplemented(self._chat_pipeline_name(req))

    def chat_stream(self, req: ChatCompletionRequest) -> Iterator[str]:
        """Validate eagerly, then return the SSE generator.

        Status code and headers are sent before the first chunk, so model
        resolution must happen here, not inside the generator.
        """
        resolve_model_dir(req.model)
        pipeline = self._chat_pipeline_name(req)

        def gen() -> Iterator[str]:
            # TODO: stream generation via a streamer callback, emitting
            # chat.completion.chunk objects: role first, then content deltas,
            # then finish_reason — and close with [DONE]. Until then: a single
            # error event in OpenAI stream-error format, then [DONE].
            err = PipelineNotImplemented(pipeline)
            yield f"data: {json.dumps(error_body(err))}\n\n"
            yield "data: [DONE]\n\n"

        return gen()

    def _chat_pipeline_name(self, req: ChatCompletionRequest) -> str:
        # VLM dispatch is decided by request shape; the CB flag only picks the
        # backend for the text-only LLM path (CB does not serve VLM).
        if messages_have_images(req.messages):
            return "VLMPipeline"
        if OV_USE_CONTINUOUS_BATCHING:
            return "ContinuousBatchingPipeline"
        return "LLMPipeline"

    # -- embeddings / rerank --------------------------------------------------

    def embeddings(self, req: EmbeddingsRequest) -> dict:
        # TODO: TextEmbeddingPipeline(model_dir, self.device)
        #       .embed_documents(inputs) -> {"object": "list", "data": [...]}
        resolve_model_dir(req.model)
        raise PipelineNotImplemented("TextEmbeddingPipeline")

    def rerank(self, req: RerankRequest) -> dict:
        # TODO: TextRerankPipeline(model_dir, self.device)
        #       .rerank(req.query, req.documents) -> {"results": [...]}
        resolve_model_dir(req.model)
        raise PipelineNotImplemented("TextRerankPipeline")

    # -- images -----------------------------------------------------------------

    def image_generate(self, req: ImagesGenerationRequest) -> dict:
        # TODO: Text2ImagePipeline(model_dir, self.device)
        #       .generate(req.prompt, ...) -> b64_json data list
        resolve_model_dir(req.model)
        raise PipelineNotImplemented("Text2ImagePipeline")

    def image_edit(
        self,
        model: str,
        prompt: str,
        image: bytes,
        mask: bytes | None,
        n: int = 1,
        size: str | None = None,
        response_format: str = "b64_json",
    ) -> dict:
        # TODO: mask present -> InpaintingPipeline(model_dir, self.device)
        #       no mask      -> Image2ImagePipeline(model_dir, self.device)
        resolve_model_dir(model)
        if not mask:
            # Zero-length upload counts as "no mask" to avoid silently picking
            # the inpainting path with an empty mask tensor.
            mask = None
        pipeline = "InpaintingPipeline" if mask is not None else "Image2ImagePipeline"
        raise PipelineNotImplemented(pipeline)

    # -- audio / video ------------------------------------------------------------

    def transcribe(
        self,
        model: str,
        audio: bytes,
        language: str | None = None,
        prompt: str | None = None,
        response_format: str = "json",
        temperature: float = 0.0,
    ) -> dict:
        # TODO: WhisperPipeline(model_dir, self.device).generate(pcm, ...)
        #       -> {"text": ...}
        resolve_model_dir(model)
        raise PipelineNotImplemented("WhisperPipeline")

    def translate(
        self,
        model: str,
        audio: bytes,
        prompt: str | None = None,
        response_format: str = "json",
        temperature: float = 0.0,
    ) -> dict:
        # TODO: WhisperPipeline with task="translate" -> {"text": ...}
        resolve_model_dir(model)
        raise PipelineNotImplemented("WhisperPipeline")

    def speech(self, req: SpeechRequest) -> bytes:
        # TODO: Text2SpeechPipeline(model_dir, self.device)
        #       .generate(req.input, ...) -> WAV bytes (route wraps audio/wav)
        resolve_model_dir(req.model)
        raise PipelineNotImplemented("Text2SpeechPipeline")

    def video_generate(self, req: VideoGenerationRequest) -> dict:
        # TODO: Text2VideoPipeline(model_dir, self.device).generate(req.prompt, ...)
        resolve_model_dir(req.model)
        raise PipelineNotImplemented("Text2VideoPipeline")


# --- 4. FastAPI app and routes -------------------------------------------------

app = FastAPI(title="openvino-genai OpenAI-compatible server", version="0.1.0")

backend = PipelineBackend(OV_MODELS_ROOT, OV_DEVICE)


@app.exception_handler(APIError)
async def api_error_handler(_: Request, exc: APIError) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content=error_body(exc))


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}


@app.get("/v1/models")
async def models() -> dict:
    return {
        "object": "list",
        "data": [
            {
                "id": m,
                "object": "model",
                "created": _model_created(m),
                "owned_by": "local",
            }
            for m in list_models()
        ],
    }


@app.post("/v1/chat/completions")
async def chat_completions(req: ChatCompletionRequest):
    if req.stream:
        return StreamingResponse(
            backend.chat_stream(req), media_type="text/event-stream"
        )
    return backend.chat(req)


@app.post("/v1/embeddings")
async def embeddings(req: EmbeddingsRequest):
    return backend.embeddings(req)


@app.post("/v1/rerank")
async def rerank(req: RerankRequest):
    return backend.rerank(req)


@app.post("/v1/images/generations")
async def images_generations(req: ImagesGenerationRequest):
    return backend.image_generate(req)


@app.post("/v1/images/edits")
async def images_edits(
    model: str = Form(...),
    prompt: str = Form(...),
    image: UploadFile = File(...),
    mask: UploadFile | None = File(None),
    n: int = Form(1),
    size: str | None = Form(None),
    response_format: str = Form("b64_json"),
):
    image_bytes = await image.read()
    mask_bytes = await mask.read() if mask is not None else None
    return backend.image_edit(
        model, prompt, image_bytes, mask_bytes, n, size, response_format
    )


@app.post("/v1/audio/transcriptions")
async def audio_transcriptions(
    model: str = Form(...),
    file: UploadFile = File(...),
    language: str | None = Form(None),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
):
    return backend.transcribe(
        model, await file.read(), language, prompt, response_format, temperature
    )


@app.post("/v1/audio/translations")
async def audio_translations(
    model: str = Form(...),
    file: UploadFile = File(...),
    prompt: str | None = Form(None),
    response_format: str = Form("json"),
    temperature: float = Form(0.0),
):
    return backend.translate(
        model, await file.read(), prompt, response_format, temperature
    )


@app.post("/v1/audio/speech")
async def audio_speech(req: SpeechRequest):
    wav = backend.speech(req)  # stub: raises; real impl returns WAV bytes
    return Response(content=wav, media_type="audio/wav")


@app.post("/v1/videos")
async def videos(req: VideoGenerationRequest):
    return backend.video_generate(req)


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=OV_SERVER_PORT)
