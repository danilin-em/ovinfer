# VLMPipeline → POST /v1/chat/completions — дизайн

Дата: 2026-06-10. Статус: одобрен.

Реализация `og.VLMPipeline` в ovinfer: заглушка 501 для chat-запросов с
изображениями заменяется рабочим инференсом. Контракт эндпоинта — из спеки
2026-06-06-openai-server-design.md (диспетчеризация по image-частям в
`messages`); детали диспетчеризации и обработки изображений фиксируются здесь.

## Ключевые факты об og (2026.2.0)

- `og.VLMPipeline(model_dir, device)`;
  `generate(prompt: str, images: list[ov.Tensor], generation_config, streamer)
  -> VLMDecodedResults`. У `VLMDecodedResults` есть `texts` и `perf_metrics` —
  существующий `Backend._chat_result` переиспользуется без изменений.
- Изображение — `ov.Tensor` из uint8 RGB HWC numpy-массива (PIL → `np.array`).
- В промпте поддерживаются теги `<ov_genai_image_i>` (i — zero-based индекс в
  списке `images`); без тегов og сам подставляет их в начало промпта.
- `GenerationConfig.apply_chat_template=True` по умолчанию: og применяет
  chat template к промпту сам. Наш путь рендерит шаблон вручную, поэтому
  для VLM выставляется `apply_chat_template=False`.
- Нативный чат-режим og (`start_chat()`/`generate()` по ходам) не поддерживает
  ссылки на изображения из предыдущих ходов («Reference to images used in
  previous prompts isn't implemented») — не подходит для stateless-сервера.

## Построение промпта (выбранный подход)

Вся история рендерится через `tokenizer.apply_chat_template` (как в LLM-пути).
Image-части сообщений заменяются тегами `<ov_genai_image_N>` внутри текста
сообщения; нумерация сквозная по всей истории в порядке появления. Все
изображения собираются в один список тензоров и передаются одним вызовом
`generate(prompt, images=tensors, generation_config=cfg)` с
`cfg.apply_chat_template = False`. Картинки работают в любом сообщении
истории — честная семантика OpenAI API.

Отвергнутые альтернативы: довериться шаблону og (история склеивается в одно
user-сообщение); replay через `start_chat()` (картинки в ранних ходах не
работают, см. выше).

## Диспетчеризация (backend.py + registry.py)

- Тип модели определяется по файлам: наличие
  `openvino_vision_embeddings_model.xml` в каталоге модели ⇒ VLM.
  Хелпер живёт в `registry.py` (знание о раскладке файлов на диске).
- VLM-модель обслуживает **и text-only запросы** через `VLMPipeline`
  (штатно работает с пустым списком изображений): у VLM-экспортов нет
  `openvino_model.xml`, `og.LLMPipeline` на них падает. Один пайплайн в кеше
  на модель, kind `"vlm"`.
- Запрос с изображениями к не-VLM модели → 400 (понятная ошибка вместо
  падения загрузки пайплайна).
- `_check_chat_supported`: ветка `PipelineNotImplemented("VLMPipeline")`
  удаляется; `use_continuous_batching` → 501 остаётся.
- `_create_pipeline`: `kind == "vlm"` → `og.VLMPipeline(model_dir, device)`.

## Изображения (backend.py)

Поддерживаемые формы image-частей (обе встречаются у OpenAI-клиентов):

- `{"type": "image_url", "image_url": {"url": "..."}}` — Chat Completions;
- `{"type": "input_image", "image_url": "..."}` — Responses-стиль, URL строкой.

Источники URL:

- `data:` URL — base64 декодируется локально;
- `http(s)` URL — сервер качает `urllib.request` с таймаутом (30 с);
- прочие схемы → 400.

Декодирование: PIL → `convert("RGB")` → `np.array` → `ov.Tensor`. Новая
зависимость: `pillow` (numpy уже приходит с openvino).

Ошибки (кривой base64, недоступный URL, не-картинка, неподдерживаемая схема,
изображения для не-VLM модели) → новый доменный класс `InvalidImage`
(errors.py) → 400 `invalid_request_error` / `invalid_image` в `_ERROR_MAP`.
Поле `detail` OpenAI игнорируется.

## Стриминг (backend.py)

Общий worker/queue-механизм `_chat_stream_gen` параметризуется вызовом
генерации: LLM — `pipe.generate([prompt], cfg, on_token)`; VLM —
`pipe.generate(prompt, images=tensors, generation_config=cfg,
streamer=on_token)`. Семантика отмены (STOP на границе токена,
`_CANCEL_JOIN_TIMEOUT`) и SSE-контракт транспорта не меняются.
Доменные ошибки (404/400/501) поднимаются до отправки заголовков, как сейчас.

## NPU-preflight (npu.py)

Kind `"vlm"` не флагуется (`reject_reason` → None): эвристика по контракту
отмечает только эмпирически проверенные падения, VLM на NPU не проверялся
(og кладёт на NPU только языковую часть VLM).

## Транспорт (server.py)

Роут `/v1/chat/completions` не меняется — messages уже передаются ядру как
есть. Обновляется docstring-карта модуля:
`VLMPipeline -> POST /v1/chat/completions (implemented, image parts)`.

## Граничные случаи

- Text-only запрос к VLM-модели → работает через VLMPipeline (пустой список
  изображений).
- Изображения к не-VLM модели → 400 `invalid_image`.
- Несуществующая модель → 404 до скачивания/декодирования изображений.
- Несколько изображений в одном и в разных сообщениях → сквозная нумерация
  тегов, один список тензоров.
- `data:` URL с не-base64 содержимым, битый JPEG, таймаут http → 400.

## Тестирование

- **Юнит** (`tests/test_vlm.py`): сборка промпта с тегами из multi-turn
  истории (сквозная нумерация, text-части без изменений); разбор обеих форм
  image-частей; декодирование `data:` URL в тензор; ошибки → `InvalidImage`.
- **Контракт** (TestClient + стаб-Backend): диспетчеризация text-only/VLM;
  400-кейсы (плохой URL, не-VLM модель); стриминг с изображением — тот же
  SSE-контракт.
- **Приёмка на реальной модели**: `OpenVINO/gemma-3-4b-it-int4-cw-ov`
  (уже скачана); сервер через `openvino-env`; curl: описание тестовой
  картинки (data: URL и http URL), multi-turn «а что на второй картинке?»,
  `stream: true`.

## Вне скоупа

- Видео (`videos`-аргумент generate), `detail`-поле OpenAI, лимиты размера
  и числа изображений, `ContinuousBatchingPipeline`.
