# ovinfer

Self-contained [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai)
inference server with an OpenAI-compatible API.

It implements LLM and vision chat (streaming and non-streaming), text
embeddings, text rerank (`POST /v1/rerank`, Jina-style response), image
generation and edits, and audio transcription, translation, and speech.
`POST /v1/videos` responds 501 until its pipeline lands.

## Prerequisites

- [OpenVINO GenAI](https://github.com/openvinotoolkit/openvino.genai) runtime
  (`openvino-genai` Python package) — see the
  [installation guide](https://docs.openvino.ai/2025/get-started/install-openvino/install-openvino-genai.html).
- Pre-converted [OpenVINO IR](https://docs.openvino.ai/2025/documentation/openvino-ir-format.html)
  model directories. Ready-to-use models are available on the
  [OpenVINO organisation on Hugging Face](https://huggingface.co/OpenVINO).

## Install & run

```bash
uv tool install <path-or-git-url>
ovinfer server --listen 127.0.0.1:8000
```

On machines that need a vendored Intel stack, run under an environment wrapper
such as [openvino-env](https://github.com/danilin-em/openvino-env)
(e.g. `openvino-env ovinfer server`).

## Configuration

CLI arguments > `OVINFER_*` environment variables > YAML config file > defaults.

| Setting | CLI | Env | Default |
|---|---|---|---|
| Models directory | `--models-root` | `OVINFER_MODELS_ROOT` | `$XDG_DATA_HOME/ovinfer/models` |
| Config file | `--config PATH` | `OVINFER_CONFIG` | none |
| Default device | `--device` | `OVINFER_DEVICE` | `CPU` |
| Per-model devices | `--device-map MODEL=DEV[,...]` | `OVINFER_DEVICE_MAP` | empty |
| Warm-up models | `--preload MODEL[:KIND][,...]` | `OVINFER_PRELOAD` | empty |
| Pipeline cache size | — | `OVINFER_MAX_PIPELINES` | `max(2, len(preload))` |
| Continuous batching | — | `OVINFER_USE_CONTINUOUS_BATCHING` | `0` (the CB path answers 501 in v0.1) |

A YAML config file supplies the same settings as a readable baseline; env vars
and CLI flags override it. Point at it with `--config PATH` or `OVINFER_CONFIG`:

```yaml
default_device: CPU
device_map:
  OpenVINO/gemma-3-4b-it-int4-cw-ov: NPU
preload:
  OpenVINO/qwen2.5-14b-instruct-int4-ov: llm
  OpenVINO/Qwen3-Embedding-0.6B-int8-ov: embedding
```

`listen` (host:port) is not read from the file — pass `--listen` instead.

Models live as `<models-root>/<org>/<model>`
[OpenVINO IR](https://docs.openvino.ai/2025/documentation/openvino-ir-format.html)
directories; the OpenAI `model` field is the `org/model` relative path.

## Example

```bash
ovinfer server \
  --device GPU \
  --device-map "OpenVINO/gemma-3-4b-it-int4-cw-ov=NPU" \
  --preload "OpenVINO/qwen2.5-14b-instruct-int4-ov,OpenVINO/Qwen3-Embedding-0.6B-int8-ov:embedding"
```

Models such as `OpenVINO/qwen2.5-14b-instruct-int4-ov` can be downloaded from the
[OpenVINO organisation on Hugging Face](https://huggingface.co/OpenVINO).

Any OpenAI client works:

```python
from openai import OpenAI

client = OpenAI(base_url="http://127.0.0.1:8000/v1", api_key="unused")
client.chat.completions.create(model="OpenVINO/qwen2.5-14b-instruct-int4-ov",
                               messages=[{"role": "user", "content": "Hi"}])
```
