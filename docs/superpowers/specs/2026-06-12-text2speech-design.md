# Text2SpeechPipeline → POST /v1/audio/speech — дизайн

Дата: 2026-06-12. Статус: одобрен, к реализации.

Реализация `og.Text2SpeechPipeline` в ovinfer: заглушка `501 Text2SpeechPipeline`
в `backend.speech` (`POST /v1/audio/speech`) заменяется рабочим TTS-инференсом.
Контракт эндпоинта — из спеки 2026-06-06-openai-server-design.md; роут, схема
`SpeechRequest` и стаб **уже есть** (`server.py`/`backend.py`, со времён
v0.1-каркаса). Эта итерация наполняет стаб, добавляет слой кодирования аудио
(samples → контейнер/кодек) и маркер TTS-модели в registry.

«Обратный Whisper»: Whisper берёт байты аудио → samples → текст; TTS берёт текст
→ samples → байты аудио. Тот же конвейер `resolve→type-check→build→generate`,
тот же принцип «формат провода — деталь транспорта».

## Скоуп и переиспользование

Роут `/v1/audio/speech` (JSON-тело `model`/`input`/`voice`/`response_format`)
**уже на месте**: транспорт зовёт `backend.speech` и оборачивает байты в
`Response`. Эта итерация добавляет: один новый kind `"tts"`, маркер
`is_tts_model`, dataclass результата, хелпер кодирования пяти+ форматов в
транспорте. **Новых зависимостей нет** (PyAV/numpy завезены Whisper'ом).

Переиспользуются без пересоздания: кэш `_pipeline`/LRU, `_create_pipeline`,
`registry.resolve_model_dir`, паттерн «resolve→type-check→build→generate» (как у
`_audio_generate`/`image_edit`), домен-ошибки `ModelNotFound`/`UnsupportedModel`
в `_ERROR_MAP`.

## Ключевые факты об og (2026.2.0, проверено)

- `og.Text2SpeechPipeline(models_path, device, **kwargs)` — конструктор как у
  всех пайплайнов.
- `generate(text: str, speaker_embedding=None, **kwargs) ->
  Text2SpeechDecodedResults`. `text` — обычная строка (или `list[str]`); шлём
  одну строку.
- `speaker_embedding` опционален: **«если не задан для SpeechT5, используется
  7306-й вектор из `Matthijs/cmu-arctic-xvectors`»** — этот дефолтный вектор
  **вшит в `libopenvino_genai.so`** (в пакете нет .bin/.npy для него), значит
  работает офлайн, сеть не нужна. v1 его и использует (см. «Голос»).
- `Text2SpeechDecodedResults.speeches` — `list` аудио-тензоров (по одному на
  входной текст), **sample rate 16 кГц** (из докстринга og). Для одного текста
  берём `speeches[0]`.
- `SpeechGenerationConfig` несёт TTS-ручки `minlenratio`/`maxlenratio`/
  `threshold` — вне скоупа v1 (см. «Вне скоупа»).

## Маркер модели и диспетчеризация

`registry.is_tts_model(model_dir)` = наличие **`openvino_vocoder.xml`**.
SpeechT5-экспорт несёт encoder/decoder/postnet/**vocoder**/tokenizer;
LLM (`openvino_model.xml`), VLM (`…vision_embeddings…`), диффузии
(`model_index.json`) и Whisper (только `…encoder…`, без вокодера) его не имеют.
Возвращает `False` для несуществующего пути (не ошибка), как
`is_vlm_model`/`is_whisper_model`.

Не-TTS модель на `/v1/audio/speech` → **400 `UnsupportedModel`** (как
non-diffusion у images, non-whisper у audio), а не падение og на build.

**Только SpeechT5 поддерживается og.Text2SpeechPipeline** (как Text2Image — только
SD/SDXL/SD3/Flux/LCM). Kokoro (`openvino_model.xml`), OuteTTS, Qwen-TTS — другой
рантайм; `is_tts_model` для них False → 400 (как `hsuwill000/Z-Image-Turbo-ov` у
Text2Image).

### Пересечение маркеров SpeechT5 ↔ Whisper

SpeechT5 несёт *и* `openvino_encoder_model.xml` (маркер Whisper), *и*
`openvino_vocoder.xml` (маркер TTS). Диспетчеризация идёт **по эндпоинтам**
(разные роуты с разной формой запроса), поэтому при корректном использовании
двусмысленности нет:

- Whisper-модель → `/v1/audio/speech`: `is_tts_model` = False → **400
  unsupported_model** (чисто).
- SpeechT5 → `/v1/audio/transcriptions`: `is_whisper_model` = True → og
  WhisperPipeline build падает → **500** (общий build-error follow-up). Редкий
  misuse; маркер Whisper **не трогаем** (YAGNI), асимметрию документируем здесь.

## Ядро (backend.py)

Заполняется стаб `speech` (добавляется dataclass результата; `voice` остаётся в
сигнатуре):

```python
_TTS_SR = 16000

@dataclass(frozen=True)
class SpeechResult:
    samples: Any      # numpy float32, mono, ~[-1, 1]
    sample_rate: int  # 16000 (SpeechT5)

def _speech_samples(result: Any) -> Any:
    """1-D float32 waveform from Text2SpeechDecodedResults.speeches[0].

    og returns .speeches as a list of ov.Tensor (one per input text); we send a
    single text, so speeches[0] is the waveform. Like _encode_pngs, the tensor's
    backing numpy view (.data) is taken explicitly (np.array on an ov.Tensor
    wraps it as a 0-d object array). Exact shape ((1, n) vs (n,)) is flattened
    with reshape(-1); confirmed at acceptance.
    """
    import numpy as np
    import openvino as ov
    speech = result.speeches[0]
    arr = np.asarray(speech.data if isinstance(speech, ov.Tensor) else speech)
    return arr.reshape(-1).astype(np.float32)

def speech(self, model_id, text, voice=None) -> SpeechResult:
    model_dir = registry.resolve_model_dir(...)        # 404 ModelNotFound
    if not registry.is_tts_model(model_dir):           # 400 UnsupportedModel
        raise UnsupportedModel(
            f"model '{model_id}' is not a Text2Speech (SpeechT5) model "
            "and cannot synthesize speech"
        )
    with self._pipeline(model_id, "tts") as pipe:
        result = pipe.generate(text)                   # default speaker
    return SpeechResult(_speech_samples(result), _TTS_SR)
```

Порядок проверок (стоимость растёт сверху вниз — как `_audio_generate`):

1. `registry.resolve_model_dir` → 404 `ModelNotFound`.
2. `registry.is_tts_model` → 400 `UnsupportedModel` (до постройки пайплайна).
3. `with _pipeline(model_id, "tts") as pipe: result = pipe.generate(text)`.
4. собрать `SpeechResult` (samples + 16 кГц).

`voice` принимается ради совместимости с OpenAI, но **не передаётся в og** —
SpeechT5 выбирает голос через xvector-эмбеддинг, а не именованный голос; используем
вшитый дефолтный speaker. Реальный выбор голоса — follow-up (см. «Вне скоупа»).
Генерация — один блокирующий вызов; стриминга/отмены нет (как у Text2Image).

Ветка в `_create_pipeline`: `kind == "tts"` →
`og.Text2SpeechPipeline(model_dir, device)`.

## Транспорт и кодирование (server.py)

Бэкенд возвращает транспорт-агностичный `SpeechResult`; кодирование в
контейнер/кодек — в транспорте (формат провода — деталь транспорта, как
`_render_transcription` у Whisper, как data-URL у картинок).

`SpeechRequest.response_format`: `str` → `Literal["wav","pcm","mp3","opus","aac",
"flac"]` (невалидное значение → **422**, как у images). **Дефолт меняется
`wav` → `mp3`** — чтобы совпасть с дефолтом OpenAI-клиентов (ради совместимости и
выбран полный набор форматов).

```python
_TTS_MEDIA_TYPES = {
    "wav": "audio/wav", "pcm": "audio/pcm", "mp3": "audio/mpeg",
    "opus": "audio/ogg", "aac": "audio/aac", "flac": "audio/flac",
}

def _encode_speech(samples, sample_rate, response_format) -> bytes:
    """Encode a mono float32 waveform to the requested container/codec.

    wav/pcm use stdlib: float32 in [-1,1] is clipped and scaled to int16 PCM;
    wav wraps it via `wave` (RIFF header), pcm is the raw little-endian int16
    bytes with no header. mp3/opus/aac/flac are muxed via PyAV (already a
    dependency): an in-memory container is opened, an audio stream of the codec
    added, an AudioFrame fed and encoded, packets muxed and flushed. Exact
    AudioFrame plumbing (sample format s16/fltp, layout=mono, pts) confirmed at
    acceptance, like _decode_audio's frame shape.
    """

@app.post("/v1/audio/speech")
def audio_speech(req: SpeechRequest) -> Response:
    result = backend.speech(req.model, req.input, req.voice)
    audio = _encode_speech(result.samples, result.sample_rate, req.response_format)
    return Response(audio, media_type=_TTS_MEDIA_TYPES[req.response_format])
```

| `response_format` | Кодек/контейнер | Content-Type | Как |
|---|---|---|---|
| `mp3` (дефолт) | MP3 | `audio/mpeg` | PyAV |
| `wav` | PCM s16 в RIFF | `audio/wav` | stdlib `wave` |
| `pcm` | голый s16 LE 16 кГц mono | `audio/pcm` | stdlib (без заголовка) |
| `opus` | Opus в Ogg | `audio/ogg` | PyAV |
| `aac` | AAC (ADTS) | `audio/aac` | PyAV |
| `flac` | FLAC | `audio/flac` | PyAV |

- Импорты `numpy`/`av`/`wave` — лениво внутри `_encode_speech` (транспорт остаётся
  лёгким на импорт, как ленивые импорты в backend).
- `pcm` — 16 кГц 16-бит mono LE (у OpenAI 24 кГц; расхождение по частоте
  документируем, как best-effort-поля у Whisper/embeddings).
- Конвертация float32→int16: `clip([-1,1])` затем `* 32767` → `int16`.

Строка карты модуля в docstring:
`Text2SpeechPipeline -> POST /v1/audio/speech (implemented)`.

## Конфиг / NPU / ошибки / зависимости

- `config.PIPELINE_KINDS += ("tts",)` — `--preload org/model:tts` становится
  валидным; отдельный кэш-kind.
- `registry` — добавляется `is_tts_model` (по образцу `is_whisper_model`).
- `npu.py` — **без изменений**: `reject_reason` для `"tts"` проваливается в `None`
  (на NPU не проверяли, не флагуем, как VLM/Whisper/image).
- Ошибки: переиспользуются `ModelNotFound` (404) и `UnsupportedModel` (400 →
  `_ERROR_MAP`: `unsupported_model`). **Новых домен-ошибок нет.**
- `pyproject.toml` — **без изменений** (`av`/`numpy` уже есть).

## Граничные случаи

- Пустой `input` (`""`) — не валидируем (YAGNI); og отдаёт что отдаст (возможно
  крошечное/пустое аудио), кодек кодирует. Чистая generate-ошибка → 500 (общий
  follow-up).
- `voice` любой/опущен → один и тот же дефолтный голос (игнорируется).
- `response_format` опущен → `mp3`.
- Длинный текст → SpeechT5 имеет лимит длины; чанкинга нет (follow-up). Падение
  на длинном входе → 500.
- `speeches` пустой/неожиданная форма → ошибка индексации/reshape → 500 (не
  ожидается; форма проверяется на приёмке).

## Тестирование

Новый `tests/test_text2speech.py`. `og` **не** импортируется (тянет
OpenVINO-рантайм/модели): dispatch стабит `_pipeline`, транспорт стабит весь
backend. `av`/`numpy` — обычные pip-зависимости (самодостаточны), поэтому
юнит-тесты кодека используют их напрямую.

- **Юнит**: `"tts"` в `PIPELINE_KINDS` и `parse_preload`; `is_tts_model` — `True`
  при наличии `openvino_vocoder.xml`, `False` иначе. `_encode_speech` на
  синтетической синусоиде (numpy) для каждого формата → проверка magic-байтов
  (`RIFF` у wav, `fLaC` у flac, `OggS` у opus, ID3/`0xFFFB` у mp3, ADTS-синк
  `0xFFF` у aac, длина = 2·N байт у pcm) **и** round-trip-декод обратно через
  PyAV (контейнер открывается, есть аудио-дорожка); число сэмплов у wav.
- **Диспатч** (стаб `_pipeline`, реальный Backend; fake-pipe пишет вызов,
  `.speeches = [numpy-массив]` → тест остаётся og-free через ветку `isinstance`):
  запрашивается kind `"tts"`; `pipe.generate` зовётся с текстом; `voice` **не**
  уходит в og (игнор); non-tts модель → 400 `UnsupportedModel` без пайплайна;
  неизвестная модель → 404 до всего.
- **Контракт** (TestClient + стаб-Backend): каждый из шести `response_format` →
  правильный Content-Type и непустое бинарное тело; невалидный формат → 422;
  `input`/`voice` доходят до backend; дефолтный формат = mp3.

**Приёмка на реальной модели**: `llmware/speech-t5-tts-ov` **уже скачана** в
`~/.openvino/models/huggingface.co/llmware/` (маркер `openvino_vocoder.xml`
подтверждён); сервер через `openvino-env` на **запасном порту** (пользователь
держит сервер на :8000); короткий текст; проверки:

- speech → 200, по одному прогону на каждый `response_format`
  (mp3/wav/pcm/opus/aac/flac), корректные magic-байты, файл воспроизводится;
- не-TTS модель (любая LLM/Whisper) → 400 `unsupported_model`;
- дефолт без `response_format` → mp3;
- подтвердить: офлайн-дефолт голоса (без сети), форма `.speeches` и 16 кГц,
  распайка `AudioFrame` в `_encode_speech`.

## Вне скоупа (YAGNI, follow-up'ы)

- Реальный выбор голоса: `voice` → speaker-эмбеддинги (загрузка/бандлинг
  xvector-файлов, маппинг имён), voice cloning.
- TTS-ручки SpeechT5: `minlenratio`/`maxlenratio`/`threshold`, `speed`.
- Стриминг ответа (`stream: true`) — только non-streaming, как у image-пайплайнов
  (threading-паттерн из `chat_stream` готов к переиспользованию).
- Чанкинг длинного текста под лимит SpeechT5.
- Не-SpeechT5 рантаймы (Kokoro/OuteTTS/Qwen-TTS) — другой пайплайн (как Z-Image
  у Text2Image).
- Кэпы на длину текста; валидация пустого `input`.
- Чистые build/generate-ошибки несобираемой/некорректной модели → 500 (общий
  follow-up для всех пайплайнов).
