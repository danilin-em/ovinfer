# ovinfer

Self-contained OpenVINO GenAI inference server with an OpenAI-compatible API.

v0.1 implements LLM chat (streaming and non-streaming), text embeddings, and
text rerank (`POST /v1/rerank`, Jina-style response); the remaining endpoints
of the contract respond 501 until their pipelines land.

## Install & run

```bash
uv tool install <path-or-git-url>
ovinfer server --listen 127.0.0.1:8000
```

On machines that need a vendored Intel stack, run under your environment
wrapper (e.g. `openvino-env ovinfer server`).

## Configuration

CLI arguments > `OVINFER_*` environment variables > defaults.

| Setting | CLI | Env | Default |
|---|---|---|---|
| Models directory | `--models-root` | `OVINFER_MODELS_ROOT` | `$XDG_DATA_HOME/ovinfer/models` |
| Default device | `--device` | `OVINFER_DEVICE` | `CPU` |
| Per-model devices | `--device-map MODEL=DEV[,...]` | `OVINFER_DEVICE_MAP` | empty |
| Warm-up models | `--preload MODEL[:KIND][,...]` | `OVINFER_PRELOAD` | empty |
| Pipeline cache size | — | `OVINFER_MAX_PIPELINES` | `max(2, len(preload))` |
| Continuous batching | — | `OVINFER_USE_CONTINUOUS_BATCHING` | `0` (the CB path answers 501 in v0.1) |

Models live as `<models-root>/<org>/<model>` OpenVINO IR directories; the
OpenAI `model` field is the `org/model` relative path.

## Example

```bash
ovinfer server \
  --device GPU \
  --device-map "OpenVINO/gemma-3-4b-it-int4-cw-ov=NPU" \
  --preload "OpenVINO/qwen2.5-14b-instruct-int4-ov,OpenVINO/Qwen3-Embedding-0.6B-int8-ov:embedding"
```

Any OpenAI client works:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
client.chat.completions.create(model="OpenVINO/qwen2.5-14b-instruct-int4-ov",
                               messages=[{"role": "user", "content": "Hi"}])
```
