# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

ovinfer is a self-contained OpenVINO GenAI inference server that exposes an
OpenAI-compatible HTTP API (`ovinfer server`, a console script). Any OpenAI
client talks to it over HTTP. The package splits a **sync core** (`Backend`, no
FastAPI imports) from a **thin FastAPI transport**; there is no public library
API yet — the core/transport boundary keeps opening one cheap later.

Running the server imports `openvino_genai`, so it needs a working OpenVINO
GenAI runtime on the path. If your OpenVINO stack lives behind an environment
wrapper such as [openvino-env](https://github.com/danilin-em/openvino-env),
run ovinfer under it (e.g. `openvino-env ovinfer server`).

## Commands

The project uses `uv` (a committed `uv.lock`); tests are `unittest`-based and
deliberately `openvino_genai`-free (they stub the pipeline layer), so they run
in ~3s without any OpenVINO environment.

```bash
# All tests (og-free, no OpenVINO env needed)
python -m unittest discover -s tests -t .

# One file / one test
python -m unittest tests.test_rerank
python -m unittest tests.test_rerank.RerankRouteTest.test_jina_response_format

# Run the server (imports openvino_genai — needs the OpenVINO env)
openvino-env ovinfer server --listen 127.0.0.1:8000

# Install the console script
uv tool install .
```

Models live as `<models-root>/<org>/<model>` OpenVINO IR directories, and the
OpenAI `model` field is the `org/model` relative path. Point
`OVINFER_MODELS_ROOT` at the directory that holds your `<org>/<model>` tree.

## Architecture

**Core / transport boundary** — the load-bearing split:

- `ovinfer/backend.py` — the sync inference core. `Backend` owns an LRU pipeline
  cache keyed by `(model_id, kind)` with a per-pipeline lock; **all**
  `openvino_genai` calls live here, and og is imported *lazily inside methods*
  so importing this module stays cheap and works with no OpenVINO env. Methods
  return plain dataclasses (`ChatResult`, `EmbedResult`, `ImageResult`,
  `TranscriptionResult`, `SpeechResult`, …).
- `ovinfer/server.py` — the only FastAPI layer. It maps requests/results onto
  the OpenAI wire format and domain errors onto HTTP status via `_ERROR_MAP`.
  Routes are plain sync `def` (FastAPI runs them in a threadpool) so blocking
  generation never stalls the event loop and `/health` stays responsive.
- `ovinfer/errors.py` — domain error types with **no** HTTP semantics; the
  transport owns the status mapping.

**Dispatch is by on-disk marker file, not config** (`ovinfer/registry.py`).
A model dir's kind is decided by which files it contains:

- `openvino_vision_embeddings_model.xml` → VLM (one `VLMPipeline` serves both
  text-only and image chat on `/v1/chat/completions`)
- `model_index.json` → diffusion (Text2Image / Image2Image / Inpainting)
- `openvino_encoder_model.xml` → Whisper ASR
- `openvino_vocoder.xml` → Text2Speech (SpeechT5)
- otherwise → plain LLM / embedding

`resolve_model_dir` validates the id against a directory scan (which also blocks
path traversal); `embedding_pooling` picks CLS/mean/last-token from the model's
own metadata rather than trusting og's CLS default.

**Config resolution** (`ovinfer/config.py`): explicit CLI args > `OVINFER_*` env
vars > YAML config file (`--config`/`OVINFER_CONFIG`, `load_config_file`) >
defaults. `cli.py` passes `None` down to mean "not given here, fall through."
The `OVINFER_` prefix avoids OpenVINO's own `OV_` variables. The file covers the
six `Settings` fields; `device_map`/`preload` are native maps and `preload`
requires an explicit `kind`.

**NPU preflight** (`ovinfer/npu.py`): a warn-only heuristic that predicts known
NPU failures (group-wise int4 LLMs, dynamic-shape embeddings) from on-disk
metadata via a light regex over the model XML — no openvino import. It only
warns; the caller still attempts the placement.

**Endpoint → pipeline map** lives in the `ovinfer/server.py` module docstring.
Chat, embeddings, rerank, images (generations/edits), and audio
(transcriptions/translations/speech) are implemented; `/v1/videos` answers 501.

## Language

- Code, comments, docstrings, README, commits — **English only**.
- Russian is for chat with the user only (reports, answers, explanations).
