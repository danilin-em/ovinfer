"""HTTP transport: FastAPI app exposing the OpenAI-compatible contract.

Routes are plain `def`, not `async def`: FastAPI runs them in a threadpool,
so blocking generation does not freeze the event loop and /health stays
responsive during inference. Starlette iterates the sync SSE generator via
iterate_in_threadpool on its own.

Pipeline-to-endpoint mapping (see docs/superpowers/specs/2026-06-06-openai-server-design.md):
    LLMPipeline                -> POST /v1/chat/completions          (implemented)
    VLMPipeline                -> POST /v1/chat/completions          (implemented, image parts)
    ContinuousBatchingPipeline -> same chat endpoint, alt backend    (501, env flag)
    TextEmbeddingPipeline      -> POST /v1/embeddings                (implemented)
    TextRerankPipeline         -> POST /v1/rerank                    (implemented)
    Text2ImagePipeline         -> POST /v1/images/generations        (implemented)
    Image2ImagePipeline        -> POST /v1/images/edits              (501)
    InpaintingPipeline         -> POST /v1/images/edits (mask)       (501)
    WhisperPipeline            -> POST /v1/audio/transcriptions|translations (501)
    Text2SpeechPipeline        -> POST /v1/audio/speech              (501)
    Text2VideoPipeline         -> POST /v1/videos                    (501)
"""

import base64
import json
import struct
import time
import uuid
from typing import Any, Iterator, Literal

from fastapi import FastAPI, File, Form, Request, Response, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from . import __version__
from .backend import Backend, ChatResult
from .errors import (
    InvalidImage,
    ModelNotFound,
    OvinferError,
    PipelineNotImplemented,
    UnsupportedModel,
)

# --- error mapping (the OpenAI error format is a transport detail) -----------

_ERROR_MAP: dict[type, tuple[int, str, str]] = {
    ModelNotFound: (404, "invalid_request_error", "model_not_found"),
    InvalidImage: (400, "invalid_request_error", "invalid_image"),
    UnsupportedModel: (400, "invalid_request_error", "unsupported_model"),
    PipelineNotImplemented: (501, "not_implemented", "pipeline_not_implemented"),
}
_DEFAULT_ERROR = (500, "internal_error", "internal_error")


def _error_body(message: str, error_type: str, code: str) -> dict:
    return {"error": {"message": message, "type": error_type, "code": code}}


def _map_error(exc: Exception) -> tuple[int, dict]:
    status, error_type, code = _ERROR_MAP.get(type(exc), _DEFAULT_ERROR)
    return status, _error_body(str(exc), error_type, code)


# --- OpenAI API request schemas -----------------------------------------------
# extra="allow": unknown OpenAI params (logit_bias, seed, user, ...) must not 422.

class ChatMessage(BaseModel):
    model_config = ConfigDict(extra="allow")
    role: Literal["system", "user", "assistant", "tool"]
    # Plain string or a list of content parts ({"type": "text"|"image_url", ...});
    # image content parts route the request to VLMPipeline; a non-VLM model
    # given images returns 400 invalid_image.
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
    # OpenAI also accepts pre-tokenized input (token arrays); unsupported here.
    input: str | list[str] | list[int] | list[list[int]]
    # openai-python defaults to base64 when the caller does not specify.
    encoding_format: Literal["float", "base64"] = "float"


class RerankRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    query: str
    documents: list[str | dict[str, Any]]
    top_n: int | None = Field(None, ge=1)


class ImagesGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    prompt: str
    n: int = Field(1, ge=1)
    size: str | None = None
    response_format: Literal["url", "b64_json"] = "b64_json"
    # Diffusion knobs (not in OpenAI's schema): forwarded to og when set, so
    # CFG-distilled turbo models (e.g. Z-Image-Turbo) can be tuned per request.
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    negative_prompt: str | None = None
    seed: int | None = None


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


# --- OpenAI response builders ---------------------------------------------------

def _completion_id() -> str:
    return "chatcmpl-" + uuid.uuid4().hex


def _chat_completion_body(model: str, result: ChatResult) -> dict:
    return {
        "id": _completion_id(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model,
        "choices": [
            {
                "index": 0,
                "message": {"role": "assistant", "content": result.text},
                "finish_reason": result.finish_reason,
            }
        ],
        "usage": {
            "prompt_tokens": result.prompt_tokens,
            "completion_tokens": result.completion_tokens,
            "total_tokens": result.prompt_tokens + result.completion_tokens,
        },
    }


def _sse_chat(model: str, deltas: Iterator[str]) -> Iterator[str]:
    """Render backend deltas as chat.completion.chunk SSE events.

    Chunk order is part of the contract: role -> content deltas ->
    finish_reason -> data: [DONE]. The finish_reason is the backend
    generator's return value (StopIteration.value).
    """
    completion_id = _completion_id()
    created = int(time.time())

    def chunk(delta: dict, finish_reason: str | None = None) -> str:
        body = {
            "id": completion_id,
            "object": "chat.completion.chunk",
            "created": created,
            "model": model,
            "choices": [
                {"index": 0, "delta": delta, "finish_reason": finish_reason}
            ],
        }
        return f"data: {json.dumps(body)}\n\n"

    yield chunk({"role": "assistant"})
    finish_reason = "stop"
    try:
        while True:
            try:
                delta = next(deltas)
            except StopIteration as stop:
                if stop.value is not None:
                    finish_reason = stop.value
                break
            yield chunk({"content": delta})
    except Exception as exc:
        # Headers are already out (200): render the error as an SSE event.
        _, body = _map_error(exc)
        yield f"data: {json.dumps(body)}\n\n"
        yield "data: [DONE]\n\n"
        return
    yield chunk({}, finish_reason=finish_reason)
    yield "data: [DONE]\n\n"


def _encode_embedding(vector: list[float], encoding_format: str) -> Any:
    if encoding_format == "base64":
        packed = struct.pack(f"<{len(vector)}f", *vector)
        return base64.b64encode(packed).decode("ascii")
    return vector


def _parse_size(size: str | None) -> tuple[int | None, int | None]:
    """Parse OpenAI 'WIDTHxHEIGHT' into (width, height).

    None or 'auto' -> (None, None): the backend omits width/height and og uses
    the model default. A malformed value raises ValueError (mapped to 400).
    """
    if size is None or size == "auto":
        return None, None
    width, sep, height = size.partition("x")
    if not sep:
        raise ValueError(f"invalid size {size!r}, expected WIDTHxHEIGHT")
    try:
        return int(width), int(height)
    except ValueError:
        raise ValueError(
            f"invalid size {size!r}, expected WIDTHxHEIGHT"
        ) from None


def _image_entry(png: bytes, response_format: str) -> dict:
    """One OpenAI images `data` entry. response_format='url' yields a data: URL
    (this server hosts no files); 'b64_json' yields the PNG encoded as base64."""
    b64 = base64.b64encode(png).decode("ascii")
    if response_format == "url":
        return {"url": "data:image/png;base64," + b64}
    return {"b64_json": b64}


# --- app factory ---------------------------------------------------------------

def create_app(backend: Backend) -> FastAPI:
    app = FastAPI(title="ovinfer OpenAI-compatible server", version=__version__)

    @app.exception_handler(OvinferError)
    async def domain_error_handler(_: Request, exc: OvinferError) -> JSONResponse:
        status, body = _map_error(exc)
        return JSONResponse(status_code=status, content=body)

    @app.get("/health")
    async def health() -> dict:
        # async on purpose: runs on the event loop, immune to threadpool
        # exhaustion while def routes are busy with blocking generation.
        return {"status": "ok"}

    @app.get("/v1/models")
    def models() -> dict:
        return {
            "object": "list",
            "data": [
                {
                    "id": m,
                    "object": "model",
                    "created": backend.model_created(m),
                    "owned_by": "local",
                }
                for m in backend.list_models()
            ],
        }

    @app.post("/v1/chat/completions")
    def chat_completions(req: ChatCompletionRequest) -> Any:
        messages = [m.model_dump() for m in req.messages]
        params: dict[str, Any] = {
            "temperature": req.temperature,
            "top_p": req.top_p,
            "max_tokens": (
                req.max_completion_tokens
                if req.max_completion_tokens is not None
                else req.max_tokens
            ),
        }
        if req.stream:
            # Domain errors (404/501) raise here, before headers go out.
            deltas = backend.chat_stream(req.model, messages, **params)
            return StreamingResponse(
                _sse_chat(req.model, deltas), media_type="text/event-stream"
            )
        result = backend.chat(req.model, messages, **params)
        return _chat_completion_body(req.model, result)

    @app.post("/v1/embeddings")
    def embeddings(req: EmbeddingsRequest) -> Any:
        if isinstance(req.input, str):
            texts = [req.input]
        elif req.input and all(isinstance(item, str) for item in req.input):
            texts = list(req.input)
        else:
            return JSONResponse(
                status_code=400,
                content=_error_body(
                    "Pre-tokenized input is not supported; "
                    "pass a string or a list of strings",
                    "invalid_request_error",
                    "unsupported_input",
                ),
            )
        result = backend.embed(req.model, texts)
        return {
            "object": "list",
            "model": req.model,
            "data": [
                {
                    "object": "embedding",
                    "index": i,
                    "embedding": _encode_embedding(vec, req.encoding_format),
                }
                for i, vec in enumerate(result.vectors)
            ],
            "usage": {
                "prompt_tokens": result.prompt_tokens,
                "total_tokens": result.prompt_tokens,
            },
        }

    @app.post("/v1/rerank")
    def rerank(req: RerankRequest) -> Any:
        # Cohere/Jina convention: a document is a string or {"text": ...}.
        texts: list[str] = []
        for doc in req.documents:
            if isinstance(doc, str):
                texts.append(doc)
            elif "text" in doc:
                texts.append(str(doc["text"]))
            else:
                return JSONResponse(
                    status_code=400,
                    content=_error_body(
                        "Each document must be a string or an object "
                        "with a 'text' field",
                        "invalid_request_error",
                        "unsupported_input",
                    ),
                )
        result = backend.rerank(req.model, req.query, texts, req.top_n)
        return {
            "model": req.model,
            "results": [
                {
                    "index": index,
                    "relevance_score": score,
                    # index is backend-authoritative; og always returns indices in
                    # range(len(texts)). An out-of-range index is a backend bug.
                    "document": {"text": texts[index]},
                }
                for index, score in result.results
            ],
            # og exposes no token counts for rerank; zeros are best-effort,
            # same as embeddings.
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        }

    @app.post("/v1/images/generations")
    def images_generations(req: ImagesGenerationRequest) -> Any:
        try:
            width, height = _parse_size(req.size)
        except ValueError as exc:
            return JSONResponse(
                status_code=400,
                content=_error_body(
                    str(exc), "invalid_request_error", "invalid_size"
                ),
            )
        result = backend.image_generate(
            req.model,
            req.prompt,
            n=req.n,
            width=width,
            height=height,
            num_inference_steps=req.num_inference_steps,
            guidance_scale=req.guidance_scale,
            negative_prompt=req.negative_prompt,
            seed=req.seed,
        )
        return {
            "created": int(time.time()),
            "data": [_image_entry(png, req.response_format) for png in result.images],
        }

    @app.post("/v1/images/edits")
    def images_edits(
        model: str = Form(...),
        prompt: str = Form(...),
        image: UploadFile = File(...),
        mask: UploadFile | None = File(None),
        n: int = Form(1),
        size: str | None = Form(None),
        response_format: str = Form("b64_json"),
    ) -> Any:
        image_bytes = image.file.read()
        mask_bytes = mask.file.read() if mask is not None else None
        return backend.image_edit(
            model, prompt, image_bytes, mask_bytes, n, size, response_format
        )

    @app.post("/v1/audio/transcriptions")
    def audio_transcriptions(
        model: str = Form(...),
        file: UploadFile = File(...),
        language: str | None = Form(None),
        prompt: str | None = Form(None),
        response_format: str = Form("json"),
        temperature: float = Form(0.0),
    ) -> Any:
        return backend.transcribe(
            model, file.file.read(), language, prompt, response_format, temperature
        )

    @app.post("/v1/audio/translations")
    def audio_translations(
        model: str = Form(...),
        file: UploadFile = File(...),
        prompt: str | None = Form(None),
        response_format: str = Form("json"),
        temperature: float = Form(0.0),
    ) -> Any:
        return backend.translate(
            model, file.file.read(), prompt, response_format, temperature
        )

    @app.post("/v1/audio/speech")
    def audio_speech(req: SpeechRequest) -> Response:
        wav = backend.speech(req.model, req.input, req.voice)
        return Response(content=wav, media_type="audio/wav")

    @app.post("/v1/videos")
    def videos(req: VideoGenerationRequest) -> Any:
        return backend.video_generate(
            req.model, req.prompt, req.size, req.seconds
        )

    return app
