# TextRerankPipeline → POST /v1/rerank — дизайн

Дата: 2026-06-10. Статус: одобрен.

Реализация `og.TextRerankPipeline` в ovinfer: заглушка 501 на `/v1/rerank`
заменяется рабочим инференсом. Контракт эндпоинта (запрос) — из спеки
2026-06-06-openai-server-design.md; формат ответа фиксируется здесь.

## Контракт

**Запрос** (`RerankRequest`, без изменений по полям): `model`, `query`,
`documents: list[str | dict]`, `top_n: int | None`. Новое: `top_n` валидируется
`ge=1`. Документ-dict обязан иметь ключ `"text"` (конвенция Cohere/Jina);
dict без `"text"` → 400 `unsupported_input` (та же семантика, что
pre-tokenized input в embeddings).

**Ответ** — Jina-стиль (понимают LangChain/LlamaIndex, Infinity, TEI):

```json
{
  "model": "OpenVINO/bge-reranker-base-int8-ov",
  "results": [
    {"index": 2, "relevance_score": 0.97,
     "document": {"text": "Париж — столица Франции."}},
    {"index": 0, "relevance_score": 0.12,
     "document": {"text": "Берлин — столица Германии."}}
  ],
  "usage": {"prompt_tokens": 0, "total_tokens": 0}
}
```

- `results` отсортированы по убыванию `relevance_score`; `index` — позиция
  документа в исходном запросе.
- `document.text` — эхо исходного текста по индексу (всегда возвращается,
  параметра `return_documents` нет — YAGNI).
- `usage` — нули: og не отдаёт счётчики токенов для rerank (best-effort,
  как в embeddings).
- `top_n` не задан → возвращаются **все** документы, отсортированные по скору.

## Ядро (backend.py)

- `RerankResult` dataclass: `results: list[tuple[int, float]]` — `(index,
  score)` в порядке убывания скора (порядок og сохраняется).
- `Backend.rerank(model_id, query, texts: list[str], top_n=None) ->
  RerankResult`: под per-pipeline-локом `pipe.rerank(query, texts)`, затем
  срез `[:top_n]`, если задан. Заглушка `PipelineNotImplemented` удаляется.
- `_create_pipeline`: `kind == "rerank"` → `og.TextRerankPipeline(model_dir,
  device, config)`, где `config = og.TextRerankPipeline.Config()` с
  `top_n = 2**31 - 1`.

**Ключевой нюанс:** дефолтный `Config().top_n` в og равен **3** — без явного
конфига пайплайн молча режет выдачу до трёх документов. В og `top_n` —
параметр пайплайна (фиксируется при создании, объект кешируется), в API —
параметр запроса. Поэтому og-срез обезвреживается большим значением при
создании, а per-request `top_n` режется в Python.

## NPU-preflight (npu.py)

`reject_reason`: `kind == "rerank"` получает ту же ветку, что `embedding` —
cross-encoder требует динамических шейпов, NPU требует статических, инференс
падает на запросе. Warn-only, как и весь preflight.

## Транспорт (server.py)

- Роут `/v1/rerank`: нормализация документов (`str` как есть; `dict` →
  `doc["text"]`; dict без `"text"` → 400), вызов ядра, сборка Jina-ответа
  с эхом текстов по индексу.
- Docstring-карта модуля: `TextRerankPipeline -> POST /v1/rerank (implemented)`.

## Граничные случаи

- Пустой `documents: []` → `results: []`. Резолв модели (404) — до раннего
  выхода; пайплайн не создаётся.
- `top_n > len(documents)` — безвреден: Python-срез.
- Несуществующая модель → 404 `model_not_found` (как везде).

## Тестирование

- **Юнит/контракт** (`tests/test_rerank.py`): FastAPI TestClient +
  стаб-Backend. Кейсы: формат ответа и сортировка; срез `top_n`;
  dict-документы с `"text"`; 400 на dict без `"text"`; эхо текста по
  индексу; пустой `documents`; `top_n=0` → 422.
- **Приёмка на реальной модели**: `OpenVINO/bge-reranker-base-int8-ov`
  (~300 МБ) скачивается в `~/.openvino/models/huggingface.co/OpenVINO/`;
  сервер через `openvino-env`; curl-проверки: релевантный документ получает
  максимальный скор, `top_n=1` режет выдачу до одного результата.

## Вне скоупа

- `return_documents`, batch-запросы нескольких query, токен-каунты в `usage`.
- Поддержка LLM-реранкеров (Qwen3-Reranker) — отдельная история, если og
  её вообще тянет через TextRerankPipeline.
