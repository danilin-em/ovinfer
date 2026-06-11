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
from .errors import InvalidImage, PipelineNotImplemented

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


def _load_image_tensor(url: str) -> Any:
    """Fetch and decode an image URL into an ov.Tensor (uint8 RGB, HWC)."""
    import io

    import numpy as np
    import openvino as ov
    from PIL import Image, UnidentifiedImageError

    raw = _fetch_image_bytes(url)
    try:
        pic = Image.open(io.BytesIO(raw)).convert("RGB")
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImage(f"could not decode image: {exc}") from exc
    return ov.Tensor(np.array(pic))


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
        if kind == "embedding":
            return og.TextEmbeddingPipeline(model_dir, device)
        if kind == "rerank":
            cfg = og.TextRerankPipeline.Config()
            cfg.top_n = _RERANK_TOP_N_MAX
            return og.TextRerankPipeline(model_dir, device, cfg)
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

    def _check_chat_supported(self, messages: list[dict[str, Any]]) -> None:
        if messages_have_images(messages):
            raise PipelineNotImplemented("VLMPipeline")
        if self.settings.use_continuous_batching:
            raise PipelineNotImplemented("ContinuousBatchingPipeline")

    def chat(
        self,
        model_id: str,
        messages: list[dict[str, Any]],
        *,
        temperature: float | None = None,
        top_p: float | None = None,
        max_tokens: int | None = None,
    ) -> ChatResult:
        """Blocking non-streaming chat completion."""
        self._check_chat_supported(messages)
        cfg = self._generation_config(temperature, top_p, max_tokens)
        with self._pipeline(model_id, "llm") as pipe:
            prompt = self._render_prompt(pipe, messages)
            # og returns a plain str for str input; a 1-element list yields
            # DecodedResults with texts and perf_metrics.
            result = pipe.generate([prompt], cfg)
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
        """Streaming chat: yields str deltas, returns the finish_reason.

        The finish_reason comes back as the generator's return value
        (PEP 380, StopIteration.value). Domain errors (404/501) are raised
        eagerly here, before the transport sends response headers.
        """
        self._check_chat_supported(messages)
        registry.resolve_model_dir(self.settings.models_root, model_id)
        cfg = self._generation_config(temperature, top_p, max_tokens)
        return self._chat_stream_gen(model_id, messages, cfg)

    def _chat_stream_gen(
        self, model_id: str, messages: list[dict[str, Any]], cfg: Any
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
                with self._pipeline(model_id, "llm") as pipe:
                    prompt = self._render_prompt(pipe, messages)
                    # og returns a plain str for str input; a 1-element list yields
                    # DecodedResults with texts and perf_metrics.
                    result = pipe.generate([prompt], cfg, on_token)
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

    # -- not implemented in v0.1 (501 contract) ---------------------------------

    def _resolve(self, model_id: str) -> Path:
        return registry.resolve_model_dir(self.settings.models_root, model_id)

    def image_generate(
        self,
        model_id: str,
        prompt: str,
        n: int = 1,
        size: str | None = None,
        response_format: str = "b64_json",
    ) -> dict:
        self._resolve(model_id)
        raise PipelineNotImplemented("Text2ImagePipeline")

    def image_edit(
        self,
        model_id: str,
        prompt: str,
        image: bytes,
        mask: bytes | None,
        n: int = 1,
        size: str | None = None,
        response_format: str = "b64_json",
    ) -> dict:
        self._resolve(model_id)
        if not mask:
            # Zero-length upload counts as "no mask" to avoid silently picking
            # the inpainting path with an empty mask tensor.
            mask = None
        pipeline = "InpaintingPipeline" if mask is not None else "Image2ImagePipeline"
        raise PipelineNotImplemented(pipeline)

    def transcribe(
        self,
        model_id: str,
        audio: bytes,
        language: str | None = None,
        prompt: str | None = None,
        response_format: str = "json",
        temperature: float = 0.0,
    ) -> dict:
        self._resolve(model_id)
        raise PipelineNotImplemented("WhisperPipeline")

    def translate(
        self,
        model_id: str,
        audio: bytes,
        prompt: str | None = None,
        response_format: str = "json",
        temperature: float = 0.0,
    ) -> dict:
        self._resolve(model_id)
        raise PipelineNotImplemented("WhisperPipeline")

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
