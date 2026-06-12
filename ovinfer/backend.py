"""Synchronous inference core: pipeline cache and generation methods.

All openvino-genai calls live behind Backend. The transport
(ovinfer.server) maps results and domain errors onto the OpenAI wire
format; nothing in this module imports FastAPI. openvino_genai itself is
imported lazily inside methods: importing ovinfer.backend stays cheap and
works without a configured OpenVINO environment.
"""

import base64
import binascii
import queue
import threading
from collections import OrderedDict
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterator

from . import config, npu, registry
from .errors import (
    InvalidAudio,
    InvalidImage,
    PipelineNotImplemented,
    UnsupportedModel,
)

_IMAGE_PART_TYPES = {"image_url", "input_image"}
_QUEUE_END = object()
# How long a cancelled stream waits for og to notice STOP. og only checks the
# callback at token boundaries, so a cancel during NPU compile/prefill cannot
# take effect immediately; past the timeout the daemon worker is left to
# finish and release the per-pipeline lock on its own.
_CANCEL_JOIN_TIMEOUT = 30.0
# og's default TextRerankPipeline.Config().top_n is 3: without an explicit
# config the pipeline silently truncates output. The API's top_n is
# per-request, but og fixes it at pipeline construction — so the pipeline
# is built effectively unbounded and rerank() slices in Python.
_RERANK_TOP_N_MAX = 2**31 - 1


@dataclass(frozen=True)
class ChatResult:
    text: str
    finish_reason: str  # "stop" | "length"
    prompt_tokens: int
    completion_tokens: int


@dataclass(frozen=True)
class EmbedResult:
    vectors: list[list[float]]
    prompt_tokens: int


@dataclass(frozen=True)
class RerankResult:
    # (document index, relevance score) sorted by score descending.
    results: list[tuple[int, float]]


@dataclass(frozen=True)
class ImageResult:
    images: list[bytes]  # each entry is a PNG-encoded image


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    text: str


@dataclass(frozen=True)
class TranscriptionResult:
    text: str
    language: str | None
    duration: float | None
    segments: list[Segment] | None  # None unless timestamps were requested


def messages_have_images(messages: list[dict[str, Any]]) -> bool:
    """True if any message carries an image content part (VLM request)."""
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            for part in content:
                if part.get("type") in _IMAGE_PART_TYPES:
                    return True
    return False


def _flatten_content(content: str | list[dict[str, Any]] | None) -> str:
    """Join text parts of an OpenAI message content into one string."""
    if content is None:
        return ""
    if isinstance(content, str):
        return content
    return "".join(
        part.get("text", "") for part in content if part.get("type") == "text"
    )


_IMAGE_FETCH_TIMEOUT = 30.0


def _extract_image_url(part: dict[str, Any]) -> str:
    """Pull the URL out of an image content part (dict or bare-string form)."""
    raw = part.get("image_url")
    url = raw.get("url") if isinstance(raw, dict) else raw
    if not isinstance(url, str) or not url:
        raise InvalidImage("image content part is missing an image_url")
    return url


def _vlm_history(
    messages: list[dict[str, Any]],
) -> tuple[list[dict[str, str]], list[str]]:
    """Render messages for a VLM prompt.

    Text parts are kept; each image part becomes an <ov_genai_image_N> tag,
    N being a running zero-based index over the whole conversation in part
    order. Returns the chat-template-ready history plus the image URLs in tag
    order (one entry per tag). og substitutes the tags with image tokens at
    generate() time.
    """
    history: list[dict[str, str]] = []
    urls: list[str] = []
    for msg in messages:
        content = msg.get("content")
        if content is None or isinstance(content, str):
            history.append({"role": msg["role"], "content": content or ""})
            continue
        rendered: list[str] = []
        for part in content:
            ptype = part.get("type")
            if ptype == "text":
                rendered.append(part.get("text", ""))
            elif ptype in _IMAGE_PART_TYPES:
                rendered.append(f"<ov_genai_image_{len(urls)}>")
                urls.append(_extract_image_url(part))
        history.append({"role": msg["role"], "content": "".join(rendered)})
    return history, urls


def _fetch_image_bytes(url: str) -> bytes:
    """Return raw image bytes for a data: or http(s) URL; raise InvalidImage."""
    if url.startswith("data:"):
        header, sep, payload = url.partition(",")
        if not sep:
            raise InvalidImage("malformed data: URL (missing comma)")
        if ";base64" in header:
            try:
                return base64.b64decode(payload, validate=True)
            except (binascii.Error, ValueError) as exc:
                raise InvalidImage(f"invalid base64 in data: URL: {exc}") from exc
        import urllib.parse

        return urllib.parse.unquote_to_bytes(payload)
    if url.startswith(("http://", "https://")):
        import urllib.error
        import urllib.request

        # No SSRF egress filtering: this fetch will follow a client-supplied URL
        # to any host, including loopback/private/link-local addresses (e.g. cloud
        # metadata). Deliberate for this local-first, no-auth server, where
        # fetching images from localhost/LAN is a legitimate use case; revisit if
        # the server is ever exposed beyond a trusted host. See the spec's
        # "Безопасность" note.
        try:
            with urllib.request.urlopen(url, timeout=_IMAGE_FETCH_TIMEOUT) as resp:
                return resp.read()
        except (urllib.error.URLError, OSError) as exc:
            raise InvalidImage(f"could not fetch image URL: {exc}") from exc
    raise InvalidImage(f"unsupported image URL scheme: {url[:32]!r}")


def _decode_image_tensor(raw: bytes, *, batched: bool = False) -> Any:
    """Decode raw image bytes into an ov.Tensor (uint8 RGB).

    pic.load() forces a full decode now so truncated/broken data raises here,
    not lazily downstream. Shared by the VLM path (via _load_image_tensor, after
    a URL fetch) and the image-edit path (raw multipart upload bytes).

    Returns a 3D HWC tensor by default (what og.VLMPipeline takes per image).
    With batched=True, returns a 4D NHWC (1,H,W,3) tensor: og.Image2ImagePipeline
    feeds the image to a model whose input is 4D ([?,?,?,?]) and rejects a bare
    3D tensor.
    """
    import io

    import numpy as np
    import openvino as ov
    from PIL import Image, UnidentifiedImageError

    try:
        pic = Image.open(io.BytesIO(raw))
        pic.load()  # force decode now: truncated/broken data raises OSError here
        pic = pic.convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImage(f"could not decode image: {exc}") from exc
    arr = np.array(pic)
    if batched:
        arr = arr[np.newaxis, ...]
    return ov.Tensor(arr)


def _load_image_tensor(url: str) -> Any:
    """Fetch and decode an image URL into an ov.Tensor (uint8 RGB, HWC)."""
    return _decode_image_tensor(_fetch_image_bytes(url))


def _decode_mask_tensor(raw: bytes) -> Any:
    """Decode an inpainting mask into an og mask_image tensor: 4D NHWC (1,H,W,3)
    uint8 where white (255) marks the region to repaint, black (0) the region to
    keep (og/diffusers convention).

    Hybrid input handling: an OpenAI-style RGBA mask marks the edit region with
    *transparent* pixels, so when an alpha channel is present it is used and
    inverted (alpha 0 -> 255 white -> repaint). A mask without alpha is read as
    plain luminance (white -> repaint), matching diffusers tutorials and simple
    b/w masks. The single-channel edit map is broadcast to 3 channels so the
    mask tensor matches the image tensor layout og's preprocessing expects.
    """
    import io

    import numpy as np
    import openvino as ov
    from PIL import Image, UnidentifiedImageError

    try:
        pic = Image.open(io.BytesIO(raw))
        pic.load()  # force decode now: truncated/broken data raises here
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImage(f"could not decode mask: {exc}") from exc
    if pic.mode in ("RGBA", "LA") or "transparency" in pic.info:
        alpha = np.array(pic.convert("RGBA"))[..., 3]
        edit = 255 - alpha          # transparent (alpha 0) -> white -> repaint
    else:
        edit = np.array(pic.convert("L"))  # white -> repaint
    rgb = np.repeat(edit[..., np.newaxis], 3, axis=2).astype(np.uint8)
    return ov.Tensor(rgb[np.newaxis, ...])


_TARGET_SR = 16000


def _strip_lang_token(raw: str | None) -> str | None:
    """'<|en|>' -> 'en'; None or '' -> None. Best-effort for verbose_json."""
    if not raw:
        return None
    return raw.strip("<|>") or None


def _decode_audio(raw: bytes) -> Any:
    """Decode uploaded audio bytes into a mono 16 kHz float32 numpy array.

    og.WhisperPipeline.generate wants raw speech normalized to ~[-1, 1] at
    16 kHz. PyAV (vendored ffmpeg) decodes any container/codec ffmpeg supports
    (wav/mp3/m4a/webm/flac/ogg/opus) in-process; an AudioResampler converts each
    decoded frame to mono float ('flt' is already in [-1, 1]) at 16 kHz, and the
    frames are concatenated. Broken/unsupported input raises InvalidAudio (400).
    """
    import io

    import av
    import numpy as np

    try:
        container = av.open(io.BytesIO(raw))
    except (av.error.FFmpegError, OSError, ValueError) as exc:
        raise InvalidAudio(f"could not open audio: {exc}") from exc
    try:
        if not container.streams.audio:
            raise InvalidAudio("file has no audio stream")
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="flt", layout="mono", rate=_TARGET_SR)
        chunks: list[np.ndarray] = []
        for frame in container.decode(stream):
            for resampled in resampler.resample(frame):
                chunks.append(resampled.to_ndarray().reshape(-1))
        for resampled in resampler.resample(None):  # flush the resampler tail
            chunks.append(resampled.to_ndarray().reshape(-1))
    except InvalidAudio:
        raise
    except (av.error.FFmpegError, OSError, ValueError) as exc:
        raise InvalidAudio(f"could not decode audio: {exc}") from exc
    finally:
        container.close()
    if not chunks:
        raise InvalidAudio("audio file contains no decodable samples")
    return np.concatenate(chunks).astype(np.float32)


def _encode_pngs(tensor: Any) -> list[bytes]:
    """Encode an og image tensor (uint8 RGB, (N,H,W,3) or (H,W,3)) as PNG bytes.

    og returns an ov.Tensor; np.array() would wrap it as a 0-d object array, so
    its backing numpy view (.data) is taken explicitly. A single-image result
    may arrive 3D; it is normalized to a 1-element batch.
    """
    import io

    import numpy as np
    import openvino as ov
    from PIL import Image

    arr = np.asarray(tensor.data if isinstance(tensor, ov.Tensor) else tensor)
    if arr.ndim == 3:
        arr = arr[np.newaxis, ...]
    pngs: list[bytes] = []
    for i in range(arr.shape[0]):
        buf = io.BytesIO()
        Image.fromarray(arr[i]).save(buf, format="PNG")
        pngs.append(buf.getvalue())
    return pngs


class _CacheEntry:
    __slots__ = ("pipeline", "lock")

    def __init__(self) -> None:
        self.pipeline: Any = None
        self.lock = threading.Lock()


class Backend:
    """Pipeline cache plus one method per contract endpoint.

    Unimplemented endpoints keep plain-Python signatures and raise
    PipelineNotImplemented so the 501 contract holds.
    """

    def __init__(
        self,
        *,
        models_root: Path | str | None = None,
        default_device: str | None = None,
        device_map: dict[str, str] | None = None,
        preload: list[tuple[str, str]] | None = None,
        max_pipelines: int | None = None,
        use_continuous_batching: bool | None = None,
    ) -> None:
        self.settings = config.load_settings(
            models_root=models_root,
            default_device=default_device,
            device_map=device_map,
            preload=preload,
            max_pipelines=max_pipelines,
            use_continuous_batching=use_continuous_batching,
        )
        self._cache: OrderedDict[tuple[str, str], _CacheEntry] = OrderedDict()
        self._cache_lock = threading.Lock()

    # -- registry ------------------------------------------------------------

    def list_models(self) -> list[str]:
        return registry.list_models(self.settings.models_root)

    def model_created(self, model_id: str) -> int:
        return registry.model_created(self.settings.models_root, model_id)

    # -- pipeline cache --------------------------------------------------------

    def preload_models(self) -> None:
        """Eagerly create pipelines listed in settings.preload."""
        for model_id, kind in self.settings.preload:
            with self._pipeline(model_id, kind):
                pass

    def _device_for(self, model_id: str) -> str:
        return self.settings.device_map.get(model_id, self.settings.default_device)

    def _entry(self, model_id: str, kind: str) -> _CacheEntry:
        """Look up or insert a cache entry; evict the LRU one over capacity."""
        key = (model_id, kind)
        with self._cache_lock:
            entry = self._cache.get(key)
            if entry is not None:
                self._cache.move_to_end(key)
                return entry
            entry = _CacheEntry()
            self._cache[key] = entry
            while len(self._cache) > self.settings.max_pipelines:
                # Drop the least recently used entry. The pipeline object is
                # freed by GC once the last thread holding its lock finishes;
                # a busy pipeline cannot be yanked out from under a generation.
                self._cache.popitem(last=False)
            return entry

    @contextmanager
    def _pipeline(self, model_id: str, kind: str) -> Iterator[Any]:
        """Yield the cached pipeline under its per-pipeline lock.

        Resolution happens before touching the cache (a 404 must not insert
        or evict entries). Creation happens under the per-pipeline lock but
        outside the global cache lock, so loading one model does not block
        requests to other models. A failed creation removes the entry from
        the cache: a permanently broken model must not pin an LRU slot.
        """
        model_dir = registry.resolve_model_dir(self.settings.models_root, model_id)
        entry = self._entry(model_id, kind)
        with entry.lock:
            if entry.pipeline is None:
                try:
                    entry.pipeline = self._create_pipeline(model_dir, model_id, kind)
                except BaseException:
                    with self._cache_lock:
                        if self._cache.get((model_id, kind)) is entry:
                            del self._cache[(model_id, kind)]
                    raise
            yield entry.pipeline

    def _create_pipeline(self, model_dir: Path, model_id: str, kind: str) -> Any:
        import openvino_genai as og

        device = self._device_for(model_id)
        # Warn-only NPU preflight: a clear heads-up before a known-doomed NPU
        # build, instead of a raw compiler traceback (LLM) or a request-time
        # 500 (embedding, rerank). Does not change the device or block the attempt.
        npu.warn_if_incompatible(model_id, model_dir, kind, device)
        if kind == "llm":
            return og.LLMPipeline(model_dir, device)
        if kind == "vlm":
            return og.VLMPipeline(model_dir, device)
        if kind == "embedding":
            # og's Config defaults to CLS pooling, which only suits encoder
            # models. Pick per-model (causal LMs need last-token) so unrelated
            # texts do not collapse to near-identical vectors. normalize stays
            # at og's default (True).
            cfg = og.TextEmbeddingPipeline.Config()
            cfg.pooling_type = {
                "cls": og.TextEmbeddingPipeline.PoolingType.CLS,
                "mean": og.TextEmbeddingPipeline.PoolingType.MEAN,
                "last_token": og.TextEmbeddingPipeline.PoolingType.LAST_TOKEN,
            }[registry.embedding_pooling(model_dir)]
            return og.TextEmbeddingPipeline(model_dir, device, cfg)
        if kind == "rerank":
            cfg = og.TextRerankPipeline.Config()
            cfg.top_n = _RERANK_TOP_N_MAX
            return og.TextRerankPipeline(model_dir, device, cfg)
        if kind == "text2image":
            return og.Text2ImagePipeline(model_dir, device)
        if kind == "image2image":
            return og.Image2ImagePipeline(model_dir, device)
        if kind == "inpainting":
            return og.InpaintingPipeline(model_dir, device)
        if kind == "whisper":
            return og.WhisperPipeline(model_dir, device)
        raise ValueError(f"Unknown pipeline kind: {kind!r}")

    # -- chat ------------------------------------------------------------------

    @staticmethod
    def _render_prompt(pipe: Any, messages: list[dict[str, Any]]) -> str:
        """Apply the model's own chat template to OpenAI-style messages."""
        history = [
            {"role": m["role"], "content": _flatten_content(m.get("content"))}
            for m in messages
        ]
        return pipe.get_tokenizer().apply_chat_template(
            history, add_generation_prompt=True
        )

    @staticmethod
    def _generation_config(
        temperature: float | None,
        top_p: float | None,
        max_tokens: int | None,
    ) -> Any:
        import openvino_genai as og

        cfg = og.GenerationConfig()
        cfg.max_new_tokens = max_tokens if max_tokens is not None else 512
        if temperature is not None and temperature > 0:
            cfg.do_sample = True
            cfg.temperature = temperature
            if top_p is not None:
                cfg.top_p = top_p
        # temperature == 0 or unset -> greedy. Deliberate deviation from
        # OpenAI's temperature=1 default: deterministic local inference is
        # more useful.
        return cfg

    @staticmethod
    def _chat_result(result: Any, cfg: Any) -> ChatResult:
        metrics = result.perf_metrics
        completion_tokens = metrics.get_num_generated_tokens()
        prompt_tokens = metrics.get_num_input_tokens()
        finish_reason = (
            "length" if completion_tokens >= cfg.max_new_tokens else "stop"
        )
        return ChatResult(
            text=result.texts[0],
            finish_reason=finish_reason,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

    def _check_chat_supported(self) -> None:
        if self.settings.use_continuous_batching:
            raise PipelineNotImplemented("ContinuousBatchingPipeline")

    def _resolve_chat_route(
        self, model_id: str, messages: list[dict[str, Any]]
    ) -> tuple[str, tuple[list[dict[str, str]], list[Any]] | None]:
        """Eager routing and validation shared by chat and chat_stream.

        Returns (kind, prepared). kind is "vlm" or "llm". For "vlm", prepared is
        (history, image_tensors) with images already loaded, so a bad image
        raises InvalidImage here — before any streaming response starts. Raises
        404 (unknown model), 400 (images to a non-VLM model or a bad image), or
        501 (continuous batching) before the transport sends headers.
        """
        self._check_chat_supported()
        model_dir = registry.resolve_model_dir(self.settings.models_root, model_id)
        if registry.is_vlm_model(model_dir):
            history, urls = _vlm_history(messages)
            images = [_load_image_tensor(url) for url in urls]
            return "vlm", (history, images)
        if messages_have_images(messages):
            raise InvalidImage(
                f"model '{model_id}' has no vision encoder and cannot accept images"
            )
        return "llm", None

    def _generate(
        self,
        pipe: Any,
        kind: str,
        messages: list[dict[str, Any]],
        prepared: tuple[list[dict[str, str]], list[Any]] | None,
        cfg: Any,
        on_token: Any = None,
    ) -> Any:
        """Run one generation. VLM gets a single templated prompt string plus
        image tensors with apply_chat_template disabled (we render the template
        ourselves); LLM keeps the 1-element-list form. on_token, when given, is
        the streaming callback. `messages` is used only by the LLM path
        (_render_prompt); the VLM path renders from `prepared`.

        Both paths omit the streamer argument entirely when not streaming,
        rather than passing None, so the call shape matches og's own samples.
        """
        if kind == "vlm":
            history, images = prepared
            prompt = pipe.get_tokenizer().apply_chat_template(
                history, add_generation_prompt=True
            )
            cfg.apply_chat_template = False
            if on_token is None:
                return pipe.generate(prompt, images=images, generation_config=cfg)
            return pipe.generate(
                prompt, images=images, generation_config=cfg, streamer=on_token
            )
        prompt = self._render_prompt(pipe, messages)
        if on_token is None:
            return pipe.generate([prompt], cfg)
        return pipe.generate([prompt], cfg, on_token)

    def chat(
        self,
        model_id: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        """Blocking non-streaming chat completion (LLM or VLM)."""
        kind, prepared = self._resolve_chat_route(model_id, messages)
        cfg = self._generation_config(temperature, top_p, max_tokens)
        with self._pipeline(model_id, kind) as pipe:
            result = self._generate(pipe, kind, messages, prepared, cfg)
        return self._chat_result(result, cfg)

    def chat_stream(
        self,
        model_id: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> Iterator[str]:
        """Streaming chat (LLM or VLM): yields str deltas, returns finish_reason.

        Routing/validation (404/400/501) and image loading happen eagerly here,
        before the transport sends response headers. The finish_reason comes
        back as the generator's return value (PEP 380, StopIteration.value).
        """
        kind, prepared = self._resolve_chat_route(model_id, messages)
        cfg = self._generation_config(temperature, top_p, max_tokens)
        return self._chat_stream_gen(model_id, messages, cfg, kind, prepared)

    def _chat_stream_gen(
        self,
        model_id: str,
        messages: list[dict[str, Any]],
        cfg: Any,
        kind: str,
        prepared: tuple[list[dict[str, str]], list[Any]] | None,
    ) -> Iterator[str]:
        import openvino_genai as og

        deltas: queue.Queue = queue.Queue()
        cancel = threading.Event()
        outcome: dict[str, Any] = {}

        def on_token(delta: str) -> Any:
            if cancel.is_set():
                return og.StreamingStatus.STOP
            deltas.put(delta)
            return og.StreamingStatus.RUNNING

        def worker() -> None:
            try:
                with self._pipeline(model_id, kind) as pipe:
                    result = self._generate(
                        pipe, kind, messages, prepared, cfg, on_token
                    )
                    outcome["result"] = self._chat_result(result, cfg)
            except BaseException as exc:  # propagated to the consuming thread
                outcome["error"] = exc
            finally:
                deltas.put(_QUEUE_END)

        thread = threading.Thread(
            target=worker, name=f"ovinfer-gen-{model_id}", daemon=True
        )
        thread.start()
        try:
            while True:
                item = deltas.get()
                if item is _QUEUE_END:
                    break
                yield item
        except GeneratorExit:
            # Client disconnected: ask og to stop at the next token, then wait
            # for the worker to release the device and the per-pipeline lock.
            cancel.set()
            thread.join(timeout=_CANCEL_JOIN_TIMEOUT)
            raise
        thread.join()
        if "error" in outcome:
            raise outcome["error"]
        return outcome["result"].finish_reason

    # -- embeddings ---------------------------------------------------------------

    def embed(self, model_id: str, texts: list[str]) -> EmbedResult:
        """Embed a batch of texts; vectors come back in input order.

        og does not expose token counts for the embedding pipeline, so
        prompt_tokens is 0 (best-effort per spec).
        """
        with self._pipeline(model_id, "embedding") as pipe:
            raw = pipe.embed_documents(texts)
            vectors = [[float(x) for x in vec] for vec in raw]
        return EmbedResult(vectors=vectors, prompt_tokens=0)

    # -- rerank -------------------------------------------------------------------

    def rerank(
        self,
        model_id: str,
        query: str,
        texts: list[str],
        top_n: int | None = None,
    ) -> RerankResult:
        """Rerank texts against the query; results sorted by score descending.

        Resolution runs first so an unknown model 404s even for an empty
        batch, which is answered without touching the pipeline cache.
        """
        # Resolve first: a 404 must surface even when texts is empty
        # (no pipeline call follows).
        registry.resolve_model_dir(self.settings.models_root, model_id)
        if not texts:
            return RerankResult(results=[])
        with self._pipeline(model_id, "rerank") as pipe:
            raw = pipe.rerank(query, texts)
        results = [(int(i), float(s)) for i, s in raw]
        if top_n is not None:
            results = results[:top_n]
        return RerankResult(results=results)

    # -- image generation ---------------------------------------------------------

    def image_generate(
        self,
        model_id: str,
        prompt: str,
        *,
        n: int = 1,
        width: int | None = None,
        height: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        negative_prompt: str | None = None,
        seed: int | None = None,
    ) -> ImageResult:
        """Text-to-image generation via og.Text2ImagePipeline.

        Resolution (404) and the diffusion-model check (400 UnsupportedModel)
        run before the pipeline is built. Only explicitly-set diffusion params
        are forwarded so og's own defaults are not clobbered; guidance_scale=0.0
        is forwarded (it is a meaningful value for CFG-distilled turbo models).
        """
        model_dir = registry.resolve_model_dir(self.settings.models_root, model_id)
        if not registry.is_diffusion_model(model_dir):
            raise UnsupportedModel(
                f"model '{model_id}' is not a diffusion model "
                "and cannot generate images"
            )
        kwargs: dict[str, Any] = {"num_images_per_prompt": n}
        if width is not None:
            kwargs["width"] = width
        if height is not None:
            kwargs["height"] = height
        if num_inference_steps is not None:
            kwargs["num_inference_steps"] = num_inference_steps
        if guidance_scale is not None:
            kwargs["guidance_scale"] = guidance_scale
        if negative_prompt is not None:
            kwargs["negative_prompt"] = negative_prompt
        if seed is not None:
            kwargs["rng_seed"] = seed
        with self._pipeline(model_id, "text2image") as pipe:
            tensor = pipe.generate(prompt, **kwargs)
        return ImageResult(images=_encode_pngs(tensor))

    # -- image editing ------------------------------------------------------------

    def image_edit(
        self,
        model_id: str,
        prompt: str,
        image: bytes,
        mask: bytes | None,
        *,
        n: int = 1,
        width: int | None = None,
        height: int | None = None,
        num_inference_steps: int | None = None,
        guidance_scale: float | None = None,
        negative_prompt: str | None = None,
        strength: float | None = None,
        seed: int | None = None,
    ) -> ImageResult:
        """Image editing: Image2Image (no mask) or Inpainting (mask) via og.

        Check order tracks the cost of each step: model resolution (404) first;
        the diffusion-model check (400) covers both branches (Inpainting needs a
        diffusion model too); the input image — and, for inpainting, the mask —
        are decoded (cheap, 400 on bad bytes) before the expensive pipeline
        build, so a broken upload 400s without loading the model. A non-empty
        mask routes to og.InpaintingPipeline (mask_image is the third positional
        arg); an empty (zero-length) mask upload counts as "no mask" ->
        Image2Image. Only explicitly-set diffusion params are forwarded, so og's
        own defaults (including strength=0.8) are not clobbered;
        guidance_scale/strength=0.0 pass through (meaningful values for
        CFG-distilled / preserve-input runs).
        """
        model_dir = registry.resolve_model_dir(self.settings.models_root, model_id)
        if not mask:
            # Zero-length upload counts as "no mask" to avoid silently picking
            # the inpainting path with an empty mask tensor.
            mask = None
        if not registry.is_diffusion_model(model_dir):
            raise UnsupportedModel(
                f"model '{model_id}' is not a diffusion model and cannot edit images"
            )
        image_tensor = _decode_image_tensor(image, batched=True)
        kwargs: dict[str, Any] = {"num_images_per_prompt": n}
        if width is not None:
            kwargs["width"] = width
        if height is not None:
            kwargs["height"] = height
        if num_inference_steps is not None:
            kwargs["num_inference_steps"] = num_inference_steps
        if guidance_scale is not None:
            kwargs["guidance_scale"] = guidance_scale
        if negative_prompt is not None:
            kwargs["negative_prompt"] = negative_prompt
        if strength is not None:
            kwargs["strength"] = strength
        if seed is not None:
            kwargs["rng_seed"] = seed
        if mask is not None:
            mask_tensor = _decode_mask_tensor(mask)
            with self._pipeline(model_id, "inpainting") as pipe:
                tensor = pipe.generate(prompt, image_tensor, mask_tensor, **kwargs)
        else:
            with self._pipeline(model_id, "image2image") as pipe:
                tensor = pipe.generate(prompt, image_tensor, **kwargs)
        return ImageResult(images=_encode_pngs(tensor))

    # -- not implemented in v0.1 (501 contract) ---------------------------------

    def _resolve(self, model_id: str) -> Path:
        return registry.resolve_model_dir(self.settings.models_root, model_id)

    # -- audio transcription ------------------------------------------------------

    _TIMESTAMP_FORMATS = ("verbose_json", "srt", "vtt")

    def transcribe(
        self,
        model_id: str,
        audio: bytes,
        language: str | None = None,
        prompt: str | None = None,
        response_format: str = "json",
        temperature: float = 0.0,
    ) -> TranscriptionResult:
        """Speech-to-text in the audio's own language via og.WhisperPipeline."""
        return self._audio_generate(
            model_id, audio, "transcribe", language, prompt,
            response_format, temperature,
        )

    def translate(
        self,
        model_id: str,
        audio: bytes,
        prompt: str | None = None,
        response_format: str = "json",
        temperature: float = 0.0,
    ) -> TranscriptionResult:
        """Speech-to-English translation via og.WhisperPipeline (task=translate)."""
        return self._audio_generate(
            model_id, audio, "translate", None, prompt,
            response_format, temperature,
        )

    def _audio_generate(
        self,
        model_id: str,
        audio: bytes,
        task: str,
        language: str | None,
        prompt: str | None,
        response_format: str,
        temperature: float,
    ) -> TranscriptionResult:
        """Shared core for transcribe/translate.

        Check order tracks cost (like image_edit): model resolution (404) first;
        the Whisper-model check (400 UnsupportedModel) before any decode; the
        audio is decoded (400 InvalidAudio on bad bytes) before the expensive
        pipeline build, so a broken upload 400s without loading the model.
        Generation params go as generate(**kwargs) — og maps them onto
        WhisperGenerationConfig fields, matching the LLM/image pipelines — and
        only explicitly-set params are forwarded. Timestamps (og chunks) are
        requested only for the formats that render them.
        """
        model_dir = registry.resolve_model_dir(self.settings.models_root, model_id)
        if not registry.is_whisper_model(model_dir):
            raise UnsupportedModel(
                f"model '{model_id}' is not a Whisper model "
                "and cannot transcribe audio"
            )
        samples = _decode_audio(audio)
        want_timestamps = response_format in self._TIMESTAMP_FORMATS
        kwargs: dict[str, Any] = {"task": task}
        if language is not None:
            kwargs["language"] = f"<|{language}|>"
        if prompt:
            kwargs["initial_prompt"] = prompt
        if temperature and temperature > 0:
            kwargs["do_sample"] = True
            kwargs["temperature"] = temperature
        if want_timestamps:
            kwargs["return_timestamps"] = True
        with self._pipeline(model_id, "whisper") as pipe:
            result = pipe.generate(samples, **kwargs)
        text = result.texts[0].strip() if result.texts else ""
        segments = None
        if want_timestamps:
            segments = [
                Segment(start=float(c.start_ts), end=float(c.end_ts), text=c.text)
                for c in (result.chunks or [])
            ]
        return TranscriptionResult(
            text=text,
            language=_strip_lang_token(getattr(result, "language", None)),
            duration=len(samples) / _TARGET_SR,
            segments=segments,
        )

    def speech(self, model_id: str, text: str, voice: str | None = None) -> bytes:
        self._resolve(model_id)
        raise PipelineNotImplemented("Text2SpeechPipeline")

    def video_generate(
        self,
        model_id: str,
        prompt: str,
        size: str | None = None,
        seconds: int | None = None,
    ) -> dict:
        self._resolve(model_id)
        raise PipelineNotImplemented("Text2VideoPipeline")
