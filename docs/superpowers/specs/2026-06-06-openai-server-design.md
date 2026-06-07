# Дизайн: OpenAI-совместимый HTTP-сервер поверх openvino-genai (server.py)

Дата: 2026-06-06
Статус: одобрен (дизайн-ревью по секциям пройдено); перенесён из `~/.openvino`.
Контракт API (маппинг пайплайнов, схемы, ошибки, SSE) актуален; структурные решения
(один файл `server.py`, `PipelineBackend` внутри него) заменены спекой
`2026-06-07-ovinfer-v0.1-design.md`; упомянутые здесь env-переменные `OV_*`
переименованы той же спекой в `OVINFER_*`.

## Контекст и мотивация

В репозитории есть venv с `openvino-genai 2026.2.0`, локальное хранилище моделей
`models/<org>/<model>` и wrapper `openvino-env`. Нужен HTTP-сервер с OpenAI-совместимым
API, чтобы стандартные клиенты (openai-python, LibreChat, Continue и т.п.) могли работать
с локальными моделями. На первом этапе — **полный каркас API с заглушками**: контракт
(схемы, роуты, ошибки, стриминг) финальный, инференс не реализован.

Набросок `pipeline.py` перечисляет 11 пайплайнов openvino-genai, которые должны быть
покрыты маппингом.

## Принятые решения

| Вопрос | Решение |
|---|---|
| Фреймворк | **FastAPI + uvicorn** (+ `python-multipart` для form-data). Стандарт де-факто для OpenAI-совместимых серверов; Pydantic-валидация и SSE из коробки. |
| Размещение | **Один файл `server.py` в корне.** Каталог `server/` не используется; распил на пакет — позже, при росте реализации. |
| Охват | **Все 11 пайплайнов** получают эндпоинты-заглушки, включая rerank и text2video (де-факто стандарты вне классического OpenAI API). |
| Стриминг | **SSE-каркас сразу**: `stream: true` в chat/completions отдаёт поток `chat.completion.chunk` + `data: [DONE]`. |
| Реестр моделей | **Скан `models/` + lazy load.** `/v1/models` сканирует `OV_MODELS_ROOT`; имя модели в запросе = относительный путь `org/model`; пайплайн создаётся при первом запросе и кешируется (в заглушках — только каркас). |
| Архитектура | **Реестр + бекенд-методы**: класс `PipelineBackend` со всеми будущими вызовами og.*; роуты — тонкие адаптеры HTTP ⇄ backend и не знают про og.*. |

## Маппинг пайплайнов на эндпоинты

11 пайплайнов → 8 основных эндпоинтов (9 путей, считая вариантный
`/v1/audio/translations`). OpenAI API сгруппирован по намерению пользователя,
og.* — по архитектуре модели, поэтому в трёх местах нужна диспетчеризация по форме запроса.

| og.* пайплайн | HTTP-эндпоинт | Примечание |
|---|---|---|
| `LLMPipeline` | `POST /v1/chat/completions` | Legacy `/v1/completions` сознательно не поддерживается |
| `VLMPipeline` | `POST /v1/chat/completions` | Диспетчеризация по `image_url` в `messages` |
| `ContinuousBatchingPipeline` | — | Не эндпоинт: альтернативный бекенд chat/completions, флаг `OV_USE_CONTINUOUS_BATCHING` |
| `TextEmbeddingPipeline` | `POST /v1/embeddings` | |
| `TextRerankPipeline` | `POST /v1/rerank` | Де-факто стандарт Cohere/Jina; у OpenAI эндпоинта нет |
| `Text2ImagePipeline` | `POST /v1/images/generations` | |
| `Image2ImagePipeline` | `POST /v1/images/edits` | image + prompt, без mask |
| `InpaintingPipeline` | `POST /v1/images/edits` | Диспетчеризация по наличию `mask` в form-data |
| `WhisperPipeline` | `POST /v1/audio/transcriptions`, `POST /v1/audio/translations` | |
| `Text2SpeechPipeline` | `POST /v1/audio/speech` | Бинарный ответ `audio/wav` |
| `Text2VideoPipeline` | `POST /v1/videos` | OpenAI Videos (Sora-style) |

Служебные: `GET /v1/models` (скан хранилища), `GET /health`.

## Структура server.py

Четыре блока сверху вниз:

1. **Config** — константы из env: `OV_SERVER_PORT` (8000), `OV_MODELS_ROOT`
   (`~/.openvino/models`), `OV_DEVICE` (`CPU`/`GPU`/`NPU`), `OV_USE_CONTINUOUS_BATCHING`.
2. **Pydantic-схемы** — запросы/ответы OpenAI API (`ChatCompletionRequest`,
   `EmbeddingsRequest`, `ImagesGenerationRequest`, `RerankRequest`, …) + `ErrorResponse`
   в формате ошибок OpenAI.
3. **PipelineBackend** — ленивый кеш пайплайнов `{(model, kind) → pipeline}` и
   стаб-методы, по одному на пайплайн:
   `chat()`/`chat_stream()`, `embeddings()`, `rerank()`,
   `image_generate()`, `image_edit()`, `transcribe()`/`translate()`, `speech()`,
   `video_generate()`. Каждый стаб бросает `PipelineNotImplemented("og.<Name>")`,
   в теле — комментарий-рецепт будущей реализации. Диспетчеризация VLM-vs-LLM и
   Inpainting-vs-Image2Image живёт здесь, не в роутах.
4. **FastAPI-роуты** — тонкие адаптеры + `/v1/models`, `/health`, exception-handler
   `PipelineNotImplemented → 501`.

## Поведение

**Семантика заглушек.** `PipelineNotImplemented` → HTTP 501 в формате ошибок OpenAI:

```json
{"error": {"message": "og.LLMPipeline backend is not implemented yet",
           "type": "not_implemented", "code": "pipeline_not_implemented"}}
```

Валидация работает уже на этапе заглушек: кривой запрос → 422 (Pydantic),
несуществующая модель → 404 `model_not_found` (формат OpenAI). Контракт API
тестируем до реализации инференса.

**`GET /v1/models`.** Скан `OV_MODELS_ROOT` на глубину два уровня (`org/model`);
критерий модели — наличие `*.xml` в каталоге. Ответ — OpenAI-формат
(`{"object": "list", "data": [{"id": "OpenVINO/Qwen3-0.6B-int4-ov", ...}]}`);
`id` = относительный путь = значение поля `model` в запросах.

**SSE-каркас.** При `stream: true` chat уходит в генератор, обёрнутый в
`StreamingResponse(media_type="text/event-stream")`. Последовательность чанков
финальная (роль → контент → `finish_reason` → `data: [DONE]`); до реализации
стрим сразу отдаёт SSE-событие с ошибкой 501-семантики и `[DONE]`.

**Multipart.** `/v1/images/edits`, `/v1/audio/transcriptions|translations` принимают
`multipart/form-data` (`UploadFile`); требуется `python-multipart`.

## Зависимости и запуск

В `Makefile` к пинам добавляются `fastapi`, `uvicorn`, `python-multipart`.

```bash
openvino-env .venv/bin/python server.py   # uvicorn.run(app, port=OV_SERVER_PORT)
```

## Критерии приёмки

- `curl localhost:8000/v1/models` — список моделей из `models/` (сейчас 7).
- `POST /v1/chat/completions` с валидным телом → 501 `pipeline_not_implemented`;
  с невалидным → 422; с несуществующей моделью → 404.
- `stream: true` → корректный SSE-поток, завершающийся `data: [DONE]`.
- Все 10 путей + служебные (`/v1/models`, `/health`) видны в `/docs` (OpenAPI).

## За скоупом

- Реализация инференса (тела стаб-методов) — следующий этап.
- Аутентификация (`Authorization: Bearer`) — не проверяется.
- `ContinuousBatchingPipeline` — только флаг и место в архитектуре, без реализации.
- Тесты — роль приёмки выполняют curl-проверки из критериев выше.
