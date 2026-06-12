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
    Image2ImagePipeline        -> POST /v1/images/edits              (implemented)
    InpaintingPipeline         -> POST /v1/images/edits (mask)       (implemented)
    WhisperPipeline            -> POST /v1/audio/transcriptions|translations (implemented)
    Text2SpeechPipeline        -> POST /v1/audio/speech              (implemented)
    Text2VideoPipeline         -> POST /v1/videos                    (501)
"""

import base64
import json
import struct
import time
import uuid
from typing import Any, Iterator, Literal

from fastapi import FastAPI, File, Form, Request, Response, UploadFile
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from . import __version__
from .backend import Backend, ChatResult
from .errors import (
    InvalidAudio,
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
    InvalidAudio: (400, "invalid_request_error", "invalid_audio"),
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
    # Accepted for OpenAI compatibility but ignored: SpeechT5 selects a voice via
    # a speaker embedding, not a named voice; og's default speaker is used.
    voice: str | None = None
    # SpeechT5 outputs 16 kHz audio; the transport encodes it to the requested
    # container/codec. Default mp3 matches OpenAI clients' default.
    response_format: Literal["wav", "pcm", "mp3", "opus", "aac", "flac"] = "mp3"


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


# --- audio transcription rendering ----------------------------------------------

def _format_timestamp(seconds: float, sep: str) -> str:
    """Seconds -> 'HH:MM:SS{sep}mmm'. sep is ',' for SRT, '.' for VTT."""
    if seconds < 0:
        seconds = 0.0
    millis = int(round(seconds * 1000))
    hours, millis = divmod(millis, 3_600_000)
    minutes, millis = divmod(millis, 60_000)
    secs, millis = divmod(millis, 1000)
    return f"{hours:02d}:{minutes:02d}:{secs:02d}{sep}{millis:03d}"


def _segments_to_srt(segments: list) -> str:
    """Render segments as SubRip (SRT): 1-based index, comma decimal separator."""
    blocks = []
    for i, seg in enumerate(segments, start=1):
        start = _format_timestamp(seg.start, ",")
        end = _format_timestamp(seg.end, ",")
        blocks.append(f"{i}\n{start} --> {end}\n{seg.text.strip()}\n")
    return "\n".join(blocks)


def _segments_to_vtt(segments: list) -> str:
    """Render segments as WebVTT: 'WEBVTT' header, dot decimal separator."""
    blocks = ["WEBVTT", ""]
    for seg in segments:
        start = _format_timestamp(seg.start, ".")
        end = _format_timestamp(seg.end, ".")
        blocks.append(f"{start} --> {end}\n{seg.text.strip()}\n")
    return "\n".join(blocks)


def _verbose_json_body(result: Any, task: str) -> dict:
    """OpenAI verbose_json body. Analytics fields og does not provide
    (avg_logprob, no_speech_prob, ...) are omitted; clients read start/end/text."""
    segments = result.segments or []
    return {
        "task": task,
        "language": result.language,
        "duration": result.duration,
        "text": result.text,
        "segments": [
            {"id": i, "seek": 0, "start": seg.start, "end": seg.end,
             "text": seg.text}
            for i, seg in enumerate(segments)
        ],
    }


def _render_transcription(result: Any, response_format: str, task: str) -> Any:
    """Render a TranscriptionResult per response_format. json/verbose_json return
    a dict (JSON); text/srt/vtt return a plain-text Response with the right
    media type."""
    if response_format == "text":
        return PlainTextResponse(result.text)
    if response_format == "srt":
        return Response(
            content=_segments_to_srt(result.segments or []),
            media_type="application/x-subrip",
        )
    if response_format == "vtt":
        return Response(
            content=_segments_to_vtt(result.segments or []),
            media_type="text/vtt",
        )
    if response_format == "verbose_json":
        return _verbose_json_body(result, task)
    return {"text": result.text}


# --- audio speech encoding ------------------------------------------------------

_TTS_MEDIA_TYPES = {
    "wav": "audio/wav",
    "pcm": "audio/pcm",
    "mp3": "audio/mpeg",
    "opus": "audio/ogg",
    "aac": "audio/aac",
    "flac": "audio/flac",
}

# (container format, codec) for the PyAV-encoded formats. wav/pcm use stdlib.
_TTS_PYAV_CODECS = {
    "mp3": ("mp3", "libmp3lame"),
    "opus": ("ogg", "libopus"),
    "aac": ("adts", "aac"),
    "flac": ("flac", "flac"),
}


def _to_int16(samples: Any) -> Any:
    """Float32 waveform in ~[-1, 1] -> clipped little-endian int16 PCM."""
    import numpy as np

    return (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")


def _encode_speech(samples: Any, sample_rate: int, response_format: str) -> bytes:
    """Encode a mono float32 waveform to the requested container/codec.

    wav/pcm use stdlib: float is scaled to int16 PCM; wav adds a RIFF header,
    pcm is the raw little-endian int16 bytes with no header. mp3/opus/aac/flac
    are muxed in-memory via PyAV (already a dependency): an AudioResampler
    converts the s16 frame to the codec's sample format, frames are encoded and
    the packets muxed, then the encoder is flushed.
    """
    import io
    import wave

    pcm = _to_int16(samples)
    if response_format == "pcm":
        return pcm.tobytes()
    if response_format == "wav":
        buf = io.BytesIO()
        with wave.open(buf, "wb") as w:
            w.setnchannels(1)
            w.setsampwidth(2)
            w.setframerate(sample_rate)
            w.writeframes(pcm.tobytes())
        return buf.getvalue()

    import av

    container_format, codec = _TTS_PYAV_CODECS[response_format]
    buf = io.BytesIO()
    container = av.open(buf, mode="w", format=container_format)
    try:
        stream = container.add_stream(codec, rate=sample_rate)
        frame = av.AudioFrame.from_ndarray(
            pcm.reshape(1, -1), format="s16", layout="mono"
        )
        frame.sample_rate = sample_rate
        resampler = av.AudioResampler(
            format=stream.format.name, layout="mono", rate=sample_rate
        )
        for rframe in resampler.resample(frame):
            for packet in stream.encode(rframe):
                container.mux(packet)
        for rframe in resampler.resample(None):  # flush the resampler tail
            for packet in stream.encode(rframe):
                container.mux(packet)
        for packet in stream.encode(None):  # flush the encoder
            container.mux(packet)
    finally:
        container.close()
    return buf.getvalue()


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
        n: int = Form(1, ge=1),
        size: str | None = Form(None),
        response_format: Literal["url", "b64_json"] = Form("b64_json"),
        # Diffusion knobs (not in OpenAI's edits schema): on multipart there is
        # no extra="allow", so each is declared explicitly and forwarded when set.
        num_inference_steps: int | None = Form(None),
        guidance_scale: float | None = Form(None),
        negative_prompt: str | None = Form(None),
        strength: float | None = Form(None),
        seed: int | None = Form(None),
    ) -> Any:
        try:
            width, height = _parse_size(size)
        except ValueError as exc:
            return JSONResponse(
                status_code=400,
                content=_error_body(
                    str(exc), "invalid_request_error", "invalid_size"
                ),
            )
        image_bytes = image.file.read()
        mask_bytes = mask.file.read() if mask is not None else None
        result = backend.image_edit(
            model,
            prompt,
            image_bytes,
            mask_bytes,
            n=n,
            width=width,
            height=height,
            num_inference_steps=num_inference_steps,
            guidance_scale=guidance_scale,
            negative_prompt=negative_prompt,
            strength=strength,
            seed=seed,
        )
        return {
            "created": int(time.time()),
            "data": [
                _image_entry(png, response_format) for png in result.images
            ],
        }

    @app.post("/v1/audio/transcriptions")
    def audio_transcriptions(
        model: str = Form(...),
        file: UploadFile = File(...),
        language: str | None = Form(None),
        prompt: str | None = Form(None),
        response_format: Literal[
            "json", "text", "verbose_json", "srt", "vtt"
        ] = Form("json"),
        temperature: float = Form(0.0),
    ) -> Any:
        result = backend.transcribe(
            model, file.file.read(), language, prompt, response_format, temperature
        )
        return _render_transcription(result, response_format, "transcribe")

    @app.post("/v1/audio/translations")
    def audio_translations(
        model: str = Form(...),
        file: UploadFile = File(...),
        prompt: str | None = Form(None),
        response_format: Literal[
            "json", "text", "verbose_json", "srt", "vtt"
        ] = Form("json"),
        temperature: float = Form(0.0),
    ) -> Any:
        result = backend.translate(
            model, file.file.read(), prompt, response_format, temperature
        )
        return _render_transcription(result, response_format, "translate")

    @app.post("/v1/audio/speech")
    def audio_speech(req: SpeechRequest) -> Response:
        result = backend.speech(req.model, req.input, req.voice)
        audio = _encode_speech(result.samples, result.sample_rate, req.response_format)
        return Response(
            content=audio, media_type=_TTS_MEDIA_TYPES[req.response_format]
        )

    @app.post("/v1/videos")
    def videos(req: VideoGenerationRequest) -> Any:
        return backend.video_generate(
            req.model, req.prompt, req.size, req.seconds
        )

    return app
