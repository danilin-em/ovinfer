# Follow-up: стриминг ответа транскрипции и live-вход аудио (realtime)

Дата создания: 2026-06-12. Статус: **запланировано, не реализовано** (вынесено из
скоупа спеки `specs/2026-06-12-whisper-design.md`).

Два независимых будущих расширения аудио-ASR поверх реализованного
`og.WhisperPipeline` (`POST /v1/audio/transcriptions|translations`, batch).
Различать важно: это **разные режимы «живости»**, у них разный объём работ и
разная применимость к WhisperPipeline.

Базовый режим (реализован/в работе по whisper-спеке): записал → отправил файл
целиком → подождал → получил весь текст. Один HTTP-запрос, один ответ.

---

## Follow-up 1 — Стриминг текста ответа (`stream: true`)

**Объём: средний. Применимо к WhisperPipeline: да (с ограничением og).**

### Что

Аудио всё равно уходит **целиком** в одном multipart-запросе; по кускам
возвращается только **распознанный текст** через SSE — как chat-стриминг.
Формат событий OpenAI (Audio API):

```
data: {"type":"transcript.text.delta","delta":"Hello"}
data: {"type":"transcript.text.delta","delta":" world"}
data: {"type":"transcript.text.done","text":"Hello world"}
data: [DONE]
```

### Поддержка og

`WhisperPipeline.generate(raw_speech, config, streamer)` принимает
`streamer: Callable[[str], StreamingStatus]`. **Ограничение og: стриминг
поддержан только для аудио < 30 с и `return_timestamps=False`.** Длинное аудио
og чанкует внутри и не зовёт streamer по-токенно.

### Эскиз реализации

- **Backend**: новый генератор `transcribe_stream`/`translate_stream`,
  переиспользующий threading-паттерн из `_chat_stream_gen` (`backend.py`):
  `queue.Queue` + daemon-поток + `threading.Event` для отмены, callback
  `on_token` кладёт дельты в очередь и возвращает `StreamingStatus.RUNNING/STOP`.
  Длину аудио мы уже знаем до генерации (`len(samples)/16000` в `_decode_audio`),
  поэтому решение про fallback принимается до старта стрима.
- **Транспорт**: `response_format` остаётся `json`/`text` (стриминг
  несовместим с таймкодами → `verbose_json/srt/vtt` при `stream:true` → 400 или
  fallback). Multipart-форма получает поле `stream: bool = Form(False)`. Новый
  рендер `_sse_transcription` отдаёт события OpenAI выше, обёрнутые в
  `StreamingResponse(media_type="text/event-stream")`. Ошибки после старта
  заголовков рендерятся как SSE-событие (как в `_sse_chat`).
- **Решить при дизайне**: поведение при `stream:true` + аудио > 30 с —
  (a) тихий fallback на batch (одно `delta` = весь текст + `done`), либо
  (b) 400 с понятным сообщением. Предпочтение — (a) fallback, чтобы клиент не
  ломался; залогировать.

### Открытые вопросы

- Точный контракт событий новой Audio-стрим-схемы OpenAI (могли добавиться
  поля `logprobs`) — свериться на момент реализации.
- Отмена при дисконнекте клиента — как в chat (`GeneratorExit` → `cancel.set()`
  → `join(timeout)`).

### Готово к переиспользованию

`_chat_stream_gen` (`backend.py`) и `_sse_chat` (`server.py`) — почти готовый
шаблон; threading/queue/cancel-машинерия не пишется заново.

---

## Follow-up 2 — Live-вход аудио (realtime)

**Объём: большой (отдельный проект). Применимо к WhisperPipeline: нет напрямую.**

### Что

Клиент шлёт аудио-чанки **по мере записи** (микрофон) и получает частичные
транскрипты в реальном времени. Это модель OpenAI Realtime API (сессии
`gpt-4o-transcribe` / `whisper` поверх WebSocket), а не HTTP multipart.

### Почему это не «дописать к Whisper-слоту»

- **og.WhisperPipeline так не умеет.** `generate()` требует **весь** сегмент
  речи (строит мел-спектрограмму по всему окну). Инкрементального приёма аудио
  нет.
- Нужен слой поверх: **VAD + скользящее окно** — буферизовать входящие чанки,
  запускать Whisper на перекрывающихся окнах, склеивать/дедуплицировать
  частичные результаты (подход whisper.cpp `stream`). Это реальная DSP/буфер-
  логика, не тонкая обёртка.
- **Новая транспортная поверхность**: WebSocket-роут (FastAPI поддерживает),
  а не текущий синхронный multipart. Протокол сессий/событий OpenAI Realtime
  (`session.update`, `input_audio_buffer.append`, `...committed`,
  `conversation.item.input_audio_transcription.delta/completed`).

### Эскиз направления (для будущего brainstorming)

- Отдельный цикл brainstorming → spec → plan: это не итерация Whisper, а новый
  модуль (`realtime`-транспорт + буферизация).
- Компоненты: WebSocket-эндпоинт; кольцевой аудио-буфер; VAD (например, Silero
  VAD как OV-модель или энергетический детектор); оконный планировщик прогонов
  Whisper; дедуп частичных гипотез; обратное событие-стрим.
- Альтернатива: модель/рантайм, спроектированные под streaming-ASR (если og
  введёт streaming-ASR API — переоценить).
- Большой риск/неизвестность: задержка vs точность на перекрытии окон;
  стоимость повторных прогонов; согласование с форматом Realtime API.

### Зависимость

Можно начинать **после** Follow-up 1 и стабилизации batch-Whisper; разделяет с
ними только `_decode_audio`-семантику (моно 16 кГц float), но добавляет
инкрементальный буфер вместо разового декода файла.

---

## Где это отслеживается

- Этот файл — единый источник по обоим follow-up'ам.
- Краткий указатель — в memory проекта (`audio-streaming-realtime-followups`) и
  в `CLAUDE.md` (раздел «Следующие шаги»).
