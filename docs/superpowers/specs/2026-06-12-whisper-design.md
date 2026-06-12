# WhisperPipeline → POST /v1/audio/transcriptions + /v1/audio/translations — дизайн

Дата: 2026-06-12. Статус: одобрен, к реализации.

Реализация `og.WhisperPipeline` в ovinfer: заглушки `501 WhisperPipeline` в
`backend.transcribe`/`backend.translate` (`POST /v1/audio/transcriptions` и
`/v1/audio/translations`) заменяются рабочим ASR-инференсом. Контракт
эндпоинтов — из спеки 2026-06-06-openai-server-design.md; оба роута и их
multipart-формы **уже реализованы** (`server.py`, со времён v0.1-каркаса) и
вызывают backend-методы. Эта итерация наполняет стабы, добавляет слой
декодирования аудио и рендер пяти `response_format`.

## Скоуп и переиспользование

Маршруты `/v1/audio/transcriptions` (`file`, `model`, `language`, `prompt`,
`response_format`, `temperature`) и `/v1/audio/translations` (то же без
`language`) **уже на месте**: транспорт читает `file.file.read()` и зовёт
`backend.transcribe`/`translate`. Развилка transcribe-vs-translate — это два
разных роута (намерение зафиксировано путём), оба обслуживает **один**
`og.WhisperPipeline`; отличие — поле `cfg.task`.

Переиспользуются без пересоздания: кэш `_pipeline`/LRU, `_create_pipeline`,
`registry.resolve_model_dir`, паттерн «resolve→type-check→decode→build→generate»
(как у `image_edit`), все домен-ошибки в `_ERROR_MAP`. Эта итерация добавляет:
один новый kind `"whisper"`, хелпер декода аудио, конфиг генерации, dataclass
результата, рендер форматов в транспорте, одну новую домен-ошибку и одну
pip-зависимость.

## Ключевые факты об og (2026.2.0, проверено)

- `og.WhisperPipeline(models_path, device, **kwargs)` — конструктор как у всех
  пайплайнов.
- `generate(raw_speech_input, generation_config=None, streamer=None, **kwargs)
  -> WhisperDecodedResults`. **`raw_speech_input` — `Sequence[float]`/numpy,
  нормализованный к ~[-1, 1], моно, 16 кГц** (НЕ байты файла). numpy float32
  подходит. Длинное аудио (>30 с) og чанкует сам.
- `WhisperGenerationConfig` (проверено присваивание строк/значений):
  - `task` — строка `"transcribe"`/`"translate"`;
  - `language` — токен `"<|en|>"` (НЕ `"en"`); опущен → og сам детектит язык;
  - `initial_prompt` — строка-затравка (наш `prompt`);
  - `return_timestamps` — bool; при `True` og заполняет `.chunks`;
  - `do_sample`/`temperature` — сэмплирование (дефолт — greedy).
- `WhisperDecodedResults`: `.texts` (`list[str]`), `.chunks`
  (`list[WhisperDecodedResultChunk]` с `start_ts`/`end_ts`/`text`), `.language`,
  `.perf_metrics`, `.words`.
- Стриминг: `generate(..., streamer=...)` поддержан **только для аудио < 30 с и
  `return_timestamps=False`** — в скоуп не входит (см. «Вне скоупа»).

## Поток данных: декод аудио (главное новое звено)

og ждёт массив сэмплов, эндпоинт принимает загруженный файл (mp3/wav/m4a/webm/…).
Новый хелпер `_decode_audio(raw: bytes) -> np.ndarray` в `backend.py` через
**PyAV** (`av`) — pip-зависимость, вендорящая ffmpeg в wheel; декод в процессе,
универсальные форматы (всё, что читает ffmpeg), самодостаточно (не зависит от
системного бинарника). Выбор подтверждён пользователем; альтернативы —
ffmpeg-subprocess (нулевые pip-деп, но внешний бинарник) и soundfile/librosa
(неполное покрытие mp3/m4a/webm) — отвергнуты.

```python
_TARGET_SR = 16000

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
        stream = container.streams.audio[0]  # no audio track -> IndexError
        resampler = av.AudioResampler(format="flt", layout="mono", rate=_TARGET_SR)
        chunks: list[Any] = []
        for frame in container.decode(stream):
            for rs in resampler.resample(frame):
                chunks.append(rs.to_ndarray().reshape(-1))
        for rs in resampler.resample(None):  # flush the resampler tail
            chunks.append(rs.to_ndarray().reshape(-1))
    except (av.error.FFmpegError, IndexError, ValueError) as exc:
        raise InvalidAudio(f"could not decode audio: {exc}") from exc
    finally:
        container.close()
    if not chunks:
        raise InvalidAudio("audio file contains no decodable samples")
    return np.concatenate(chunks).astype(np.float32)
```

`format="flt"` уже даёт нормализацию к [-1, 1] — отдельной нормализации нет.
Точная форма фрейма (`to_ndarray()` для моно-`flt` → `(1, n)`; `reshape(-1)`
уплощает) и поведение `resample(None)`-flush подтверждаются на приёмке (как
Image2Image обнаружил необходимость `batched=True`).

## Конфиг генерации и диспетчеризация модели

Параметры генерации передаются как **kwargs в `generate`** (не через
сконструированный og-объект): `og.WhisperPipeline.generate(raw_speech, **kwargs)`
документированно принимает ключи, соответствующие полям
`WhisperGenerationConfig` — ровно как `LLMPipeline.generate(prompt,
max_new_tokens=...)` и все image-пайплайны (`pipe.generate(prompt, width=...)`).
Это держит юнит/диспатч-тесты **свободными от импорта og** (fake-pipe пишет dict
kwargs, тест проверяет dict), как у Image2Image/Inpainting.

`_audio_generate` собирает только заданные kwargs (по образцу `image_generate`):

- `kwargs["task"] = task` (`"transcribe"`/`"translate"`) — всегда;
- `language is not None` → `kwargs["language"] = f"<|{language}|>"` (OpenAI шлёт
  ISO-639-1 `"en"`; og ждёт `"<|en|>"`). Опущен → ключа нет → og сам детектит
  язык;
- `prompt` (непустой) → `kwargs["initial_prompt"] = prompt`;
- `temperature > 0` → `kwargs["do_sample"] = True; kwargs["temperature"] =
  temperature` (как в chat; `temperature == 0`/опущен → greedy, детерминизм);
- `return_timestamps` (True только для verbose_json/srt/vtt) →
  `kwargs["return_timestamps"] = True`; False → ключа нет (og-дефолт False).

Вызов: `pipe.generate(samples, **kwargs)`.

Маркер модели: `registry.is_whisper_model(model_dir)` = наличие
`openvino_encoder_model.xml` (Whisper-экспорт — encoder-decoder с отдельным
энкодером; LLM (`openvino_model.xml`) / VLM
(`openvino_vision_embeddings_model.xml`) его не имеют). Не-Whisper модель,
посланная на аудио-эндпоинт → **400 `UnsupportedModel`** (как non-diffusion у
images), а не падение og на build. Возвращает `False` для несуществующего пути
(не ошибка), как `is_vlm_model`/`is_diffusion_model`.

## Результат и рендер пяти форматов

Бэкенд возвращает транспорт-агностичный результат; рендер `response_format` —
в транспорте (формат проводов — деталь транспорта, как формат ошибок/картинок).

```python
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
```

`text` = `result.texts[0]` (с `.strip()`). `segments` = из `result.chunks`
(только когда `return_timestamps`), иначе `None`. `language` = `result.language`
с обрезкой `<|...|>` → `"en"` (best-effort, для verbose_json). `duration` =
`len(samples) / _TARGET_SR` (точно из декодированного массива — не зависит от
наличия таймкодов).

Рендер (`server.py`):

| `response_format` | Ответ | Content-Type |
|---|---|---|
| `json` (дефолт) | `{"text": ...}` | `application/json` |
| `text` | тело — голый текст | `text/plain` |
| `verbose_json` | `{task, language, duration, text, segments:[…]}` | `application/json` |
| `srt` | `N\nHH:MM:SS,mmm --> HH:MM:SS,mmm\ntext\n\n` | `application/x-subrip` |
| `vtt` | `WEBVTT\n\nHH:MM:SS.mmm --> HH:MM:SS.mmm\ntext\n\n` | `text/vtt` |

- `verbose_json` segment: `{id, start, end, text}` (+ `seek=0`). Аналитические
  поля OpenAI (`avg_logprob`, `no_speech_prob`, `compression_ratio`, `tokens`,
  `temperature`) og не даёт → опускаем (best-effort, как нулевые `prompt_tokens`
  у embeddings/rerank). Клиенты читают `start`/`end`/`text`.
- `task` в verbose_json: `"transcribe"`/`"translate"` (транспорт знает, какой
  роут вызван).
- Хелпер форматирования таймкода общий, различается разделителем дробной части:
  SRT — запятая (`,mmm`), VTT — точка (`.mmm`).
- Нет сегментов (таймкоды не запрошены / пустое аудио) → srt/vtt пустые,
  verbose_json с `segments: []`.

## Ядро (backend.py)

`transcribe`/`translate` — **сигнатуры не меняются** (уже в стабах), тела зовут
общий `_audio_generate`:

```python
def transcribe(self, model_id, audio, language=None, prompt=None,
               response_format="json", temperature=0.0) -> TranscriptionResult:
    return self._audio_generate(model_id, audio, "transcribe",
                                language, prompt, response_format, temperature)

def translate(self, model_id, audio, prompt=None,
              response_format="json", temperature=0.0) -> TranscriptionResult:
    return self._audio_generate(model_id, audio, "translate",
                                None, prompt, response_format, temperature)
```

Порядок проверок в `_audio_generate` (стоимость растёт сверху вниз — как
`image_edit`):

1. `registry.resolve_model_dir` → 404 `ModelNotFound`.
2. `registry.is_whisper_model` → 400 `UnsupportedModel`.
3. `_decode_audio(audio)` → 400 `InvalidAudio` (дёшево, до постройки пайплайна,
   чтобы битый upload давал 400 без загрузки модели).
4. `return_timestamps = response_format in ("verbose_json", "srt", "vtt")`.
5. собрать `kwargs` (только заданные поля; см. выше).
6. `with _pipeline(model_id, "whisper") as pipe: result = pipe.generate(samples, **kwargs)`.
7. собрать `TranscriptionResult` (text/language/duration/segments).

Генерация — один блокирующий вызов; стриминга/отмены по токенам нет (как у
Text2Image/Image2Image/Inpainting).

Ветка в `_create_pipeline`: `kind == "whisper"` →
`og.WhisperPipeline(model_dir, device)`.

## Транспорт (server.py)

Роуты `audio_transcriptions`/`audio_translations` сейчас возвращают результат
backend как dict напрямую. Меняется:

- `response_format: str = Form("json")` → `Literal["json","text","verbose_json",
  "srt","vtt"] = Form("json")` — невалидное значение даёт 422 (как у
  images_edits); это закрывает рендер на пять известных веток.
- Тело роута: получить `TranscriptionResult`, отдать его в общий рендер
  `_render_transcription(result, response_format, task=...)`, который возвращает
  `dict` (json/verbose_json) или `Response`/`PlainTextResponse` (text/srt/vtt) с
  нужным `media_type`.
- Хелперы `_segments_to_srt(segments)`, `_segments_to_vtt(segments)`,
  `_format_timestamp(seconds, sep)`.

Строки карты модуля в docstring:
`WhisperPipeline -> POST /v1/audio/transcriptions|translations (implemented)`.

## Ошибки

- Новая `InvalidAudio(OvinferError)` в `errors.py` (аудио не открылось/не
  декодировалось/без аудио-дорожки) → `_ERROR_MAP`:
  `(400, "invalid_request_error", "invalid_audio")`.
- Несуществующая модель → 404 `model_not_found` (резолв первым).
- Не-Whisper модель → 400 `invalid_request_error`/`unsupported_model`
  (`UnsupportedModel`).
- Битое/неподдерживаемое аудио → 400 `invalid_request_error`/`invalid_audio`.
- Невалидный `response_format` → 422 (`Form(..., Literal[...])`).
- Невалидный `language` (нет в `lang_to_id` модели) → ошибка og → 500 (общий
  follow-up «чистые generate-ошибки → 500», как у несобираемых диффузий).

## Конфиг / preload / NPU / зависимости

- `config.PIPELINE_KINDS += ("whisper",)` — `--preload org/model:whisper`
  становится валидным; отдельный кэш-kind.
- `registry` — добавляется `is_whisper_model` (по образцу `is_vlm_model`).
- `npu.py` — **без изменений**: `reject_reason` для `"whisper"` проваливается в
  `None` (на NPU не проверяли, не флагуем, как VLM/Text2Image/Image2Image).
- `pyproject.toml` dependencies **+ `"av"`** (PyAV). numpy уже тянется openvino.

## Граничные случаи

- Пустой/нулевой файл, файл без аудио-дорожки → `InvalidAudio` (400).
- Моно/стерео/любая частота на входе → ресемпл к моно 16 кГц (PyAV).
- `language` опущен → авто-детект og; задан → `<|{language}|>`.
- `translate` всегда в английский (og task), `language` у translate отсутствует
  в форме (как у OpenAI).
- `temperature = 0`/опущен → greedy; `> 0` → сэмплирование.
- `response_format` с таймкодами на пустом распознавании → пустые segments.
- Длинное аудио (>30 с) → og чанкует внутри; отдаём весь массив.

## Тестирование

Новый `tests/test_whisper.py`. `og` **не** импортируется (тянет
OpenVINO-рантайм/модели): dispatch стабит `_pipeline`, транспорт стабит весь
backend. `av` — обычная pip-зависимость (самодостаточный wheel, без
openvino-env), поэтому юнит-тест `_decode_audio` использует её напрямую.

- **Юнит**: `"whisper"` в `PIPELINE_KINDS` и `parse_preload`;
  `_decode_audio` на сгенерированном WAV (stdlib `wave` — без новых
  тест-зависимостей; PyAV читает WAV) → моно/16к/ожидаемая длина сэмплов; мусор
  → `InvalidAudio`; файл без аудио → `InvalidAudio`. `is_whisper_model` —
  `True` при наличии `openvino_encoder_model.xml`, `False` иначе. Рендеры:
  `_segments_to_srt`/`_segments_to_vtt`/`_format_timestamp` на синтетических
  сегментах (формат таймкода, нумерация, разделители `,`/`.`); verbose_json-тело
  из `TranscriptionResult`.
- **Диспатч** (стаб `_pipeline`, реальный Backend; fake-pipe пишет kwargs):
  запрашивается kind `"whisper"`; transcribe → `kwargs["task"]=="transcribe"`,
  translate → `"translate"`; `language="ru"` → `kwargs["language"]=="<|ru|>"`,
  опущен → ключа нет; `prompt` → `kwargs["initial_prompt"]`; `response_format` в
  {verbose_json,srt,vtt} → `kwargs["return_timestamps"]==True`, в {json,text} →
  ключа нет; `temperature>0` → `do_sample`+`temperature`, `0.0` → ключей нет;
  non-whisper модель → 400 `UnsupportedModel` без пайплайна; битое аудио → 400
  `InvalidAudio` без пайплайна; неизвестная модель → 404 до всего.
- **Контракт** (TestClient + multipart + стаб-Backend): каждый из пяти
  `response_format` → правильный content-type и форма тела (json/verbose_json —
  JSON; text/srt/vtt — текст); `language`/`prompt`/`temperature` доходят до
  backend; translate-роут не принимает `language`.

**Приёмка на реальной модели**: скачать `OpenVINO/whisper-base-fp16-ov`
(multilingual) в `~/.openvino/models/huggingface.co/OpenVINO/`; сервер через
`openvino-env` на **запасном порту** (пользователь держит сервер на :8000);
короткий аудио-сэмпл; проверки:

- transcribe → 200, по одному прогону на каждый `response_format`
  (json/text/verbose_json/srt/vtt), таймкоды непустые в verbose_json/srt/vtt;
- translate (не-английский сэмпл → английский текст) → 200;
- не-аудио модель (любая LLM) на аудио-эндпоинт → 400 `unsupported_model`;
- мусор вместо аудио → 400 `invalid_audio`;
- `language="en"` форсит язык; авто-детект без `language`.

## Вне скоупа (YAGNI)

- Стриминг транскрипции (`stream: true` → SSE `transcript.text.delta`) — только
  non-streaming, как у image-пайплайнов (отдельный follow-up; threading-паттерн
  из `chat_stream` готов к переиспользованию).
- Word-level таймкоды (`og.words`/`word_timestamps`) — только сегментные
  `chunks`.
- Диаризация, несколько аудио-файлов в запросе, `hotwords`, beam-search-ручки.
- Кэпы на размер файла / длину аудио / длину промпта.
- Аналитические поля сегмента verbose_json (`avg_logprob`, `no_speech_prob`, …)
  — og их не даёт.
- Чистые build/generate-ошибки несобираемой/некорректной модели → 500 (общий
  follow-up для всех пайплайнов).
