# Text2ImagePipeline → POST /v1/images/generations — дизайн

Дата: 2026-06-11. Статус: одобрен (дизайн-ревью по секциям пройдено).

Реализация `og.Text2ImagePipeline` в ovinfer: заглушка 501 на
`POST /v1/images/generations` заменяется рабочим инференсом. Контракт эндпоинта —
из спеки 2026-06-06-openai-server-design.md; детали маппинга параметров,
кодирования изображений и диспетчеризации фиксируются здесь.

## Ключевые факты об og (2026.2.0)

- `og.Text2ImagePipeline(models_path)` и `(models_path, device, **props)`;
  `generate(prompt: str, **kwargs) -> ov.Tensor`. Тензор результата — uint8 RGB
  батч изображений формы `(N, H, W, 3)` (N = `num_images_per_prompt`).
- Параметры `generate` (kwargs): `num_images_per_prompt`, `guidance_scale`,
  `height`, `width`, `num_inference_steps`, `negative_prompt`, `rng_seed`,
  `generator`, `prompt_2/3`, `strength`, `max_sequence_length`, `adapters`.
- **Дефолты `og.ImageGenerationConfig` конфликтуют с turbo-моделью**:
  `num_inference_steps=50`, `guidance_scale=7.5`, `rng_seed=42`, `height=-1`,
  `width=-1` (-1 = дефолт модели). Локальная модель `Z-Image-Turbo`
  дистиллирована под **8 шагов** и **guidance_scale=0.0** (README) — с дефолтами
  og она ~6× медленнее и пересвечена.
- `og` детерминирован по умолчанию (`rng_seed=42`): одинаковый prompt → одинаковая
  картинка. Совпадает с философией проекта (greedy-дефолт chat: «детерминированный
  локальный инференс полезнее»). Вариативность — через явный `seed`.

## Маппинг параметров (выбранный подход)

`/v1/images/generations` стандартного OpenAI несёт только `prompt/n/size/
response_format` — этого мало для diffusion-модели. Схема дополняется
**опциональными полями-расширениями**, которые pass-through уходят в `og`;
при отсутствии — дефолты `og`:

| Поле запроса | og kwarg | Примечание |
|---|---|---|
| `prompt` | `prompt` | обязательно |
| `n` (=1) | `num_images_per_prompt` | |
| `size` ("ШxВ") | `width`, `height` | OpenAI: ширина×высота; `None`/`"auto"` → не передаём |
| `num_inference_steps?` | `num_inference_steps` | ext.: для turbo задать 8 |
| `guidance_scale?` | `guidance_scale` | ext.: для turbo задать 0.0 |
| `negative_prompt?` | `negative_prompt` | ext. |
| `seed?` | `rng_seed` | ext.: вариативность поверх дефолтного 42 |
| `response_format` | — | wire-формат, см. ниже |

В `og.generate` передаются **только заданные** kwargs: `None`-поля не
передаются, чтобы не затирать дефолты пайплайна. `num_images_per_prompt=n`
передаётся всегда.

Отвергнутые альтернативы: авто-дефолты под модель (детект turbo / чтение
мини-конфига — лишняя «магия» и код, YAGNI); строго минимальный OpenAI-контракт
(единственная локальная модель почти неюзабельна из коробки).

## Диспетчеризация (registry.py + backend.py)

- Маршрут `/v1/images/generations` фиксирует намерение «text2image», поэтому
  диспетчеризация по форме запроса (как VLM-vs-LLM в chat) не нужна: kind жёстко
  `"text2image"`.
- Новый хелпер `registry.is_diffusion_model(model_dir)` — маркер диффузии —
  наличие `model_index.json` в каталоге модели (его нет ни у LLM-, ни у
  VLM-экспортов; знание о раскладке файлов на диске живёт в `registry.py`, как
  `is_vlm_model`). Возвращает `False` (не ошибку) для несуществующего пути.
- Не-диффузионная модель на этот маршрут → 400 (понятная ошибка вместо падения
  загрузки пайплайна).
- `_create_pipeline`: ветка `kind == "text2image"` → `og.Text2ImagePipeline(
  model_dir, device)`. Стаб `PipelineNotImplemented("Text2ImagePipeline")`
  удаляется; 501 для Image2Image/Inpainting и прочих остаётся.

## Ядро (backend.py)

Новый `@dataclass(frozen=True) ImageResult(images: list[bytes])` — список
**PNG-байтов**. Это граница ядро/транспорт: ядро кодирует PNG через PIL (уже
зависимость, уже используется для VLM-декода); транспорт делает только
base64/data-URL-обёртку (wire-формат).

```python
def image_generate(self, model_id, prompt, *, n=1, width=None, height=None,
                   num_inference_steps=None, guidance_scale=None,
                   negative_prompt=None, seed=None) -> ImageResult
```

Поток: резолв модели (404) → `is_diffusion_model` (400 `UnsupportedModel`) →
`_pipeline(model_id, "text2image")` → `pipe.generate(prompt, **kwargs)` → тензор
`(N,H,W,3)` → `_encode_pngs` (`np.array` → по срезу `Image.fromarray` → PNG в
`BytesIO`). Тензор может прийти 3D (одиночное изображение) — нормализуем к 4D.

Генерация — один блокирующий вызов. Стриминга/отмены по токенам нет (в отличие от
chat): роут `def` исполняется в threadpool Starlette, длинная генерация не
блокирует event loop, `/health` отзывчив.

## Транспорт (server.py)

Схема:

```python
class ImagesGenerationRequest(BaseModel):
    model_config = ConfigDict(extra="allow")
    model: str
    prompt: str
    n: int = 1
    size: str | None = None
    response_format: Literal["url", "b64_json"] = "b64_json"
    num_inference_steps: int | None = None
    guidance_scale: float | None = None
    negative_prompt: str | None = None
    seed: int | None = None
```

`_parse_size(size) -> (width, height)`: `"1024x768"` → `(1024, 768)`;
`None`/`"auto"` → `(None, None)`; кривой формат → 400 `invalid_request_error`/
`invalid_size` (JSONResponse в роуте, как bad-input в embeddings/rerank).

Роут вызывает `backend.image_generate(...)` и строит конверт:

```json
{"created": <ts>, "data": [{"b64_json": "..."}]}
```

Кодирование элемента: `b64_json` → base64(PNG); `url` → `"data:image/png;base64,"
+ base64(PNG)` (локальный сервер ничего не хостит; data-URL — валидный URL,
открывается в браузере/большинстве клиентов, url-дефолтящие клиенты продолжают
работать). Дефолт схемы — `b64_json`.

Карта модуля в docstring обновляется:
`Text2ImagePipeline -> POST /v1/images/generations (implemented)`.

## Ошибки

- Несуществующая модель → 404 `model_not_found` (существующий резолв, первым —
  до построения пайплайна).
- Не-диффузионная модель → новый `UnsupportedModel(OvinferError)` → 400
  `invalid_request_error`/`unsupported_model` (добавляется в `errors.py` и
  `_ERROR_MAP`).
- Кривой `size` → 400 `invalid_request_error`/`invalid_size` в транспорте.

## Конфиг / preload / NPU

- `config.PIPELINE_KINDS` += `"text2image"` — `--preload org/model:text2image`
  становится валидным (диффузию можно прогреть на старте).
- `npu.reject_reason` для `"text2image"` → `None` (падение на NPU не проверялось,
  как и VLM — не флагуем; эвристика отмечает только эмпирически проверенные
  падения). Изменений в `npu.py` не требуется (kind проваливается в `None`).

## Граничные случаи

- `size` опущен → `width/height` не передаются, работает дефолт модели/og.
- `n > 1` → тензор `(n,H,W,3)`, `data` из n элементов.
- `seed` опущен → дефолтный `rng_seed=42` (детерминированно); задан → вариативно.
- Turbo-модель без ext-полей → отработает, но медленно/пересвечено (50 шагов,
  guidance 7.5) — это осознанный trade-off (см. «Маппинг параметров»).
- `response_format="url"` → data-URL; `"b64_json"` (дефолт) → base64.

## Тестирование

- **Юнит**: `_parse_size` (валид `ШxВ`, `auto`/`None`, кривой → ошибка);
  `_encode_pngs` на фейк-тензоре (`(N,H,W,3)` numpy → N валидных PNG; нормализация
  3D→4D); `is_diffusion_model` (temp-каталог с/без `model_index.json`); сборка
  kwargs (только заданные параметры доходят до `generate`: `n`→
  `num_images_per_prompt`, `seed`→`rng_seed`, опущенные — отсутствуют).
- **Контракт** (TestClient + стаб-Backend): форма ответа `b64_json`; `url` →
  data-URL; кривой `size` → 400; проброс ext-полей в backend; не-диффузионная
  модель → 400 `unsupported_model`; несуществующая → 404.
- **Дисптач** (стаб `_pipeline`): запрашивается kind `"text2image"`;
  не-диффузионная модель → `UnsupportedModel` без построения пайплайна;
  несуществующая модель → 404 до пайплайна.
- **Приёмка на реальной модели**: `hsuwill000/Z-Image-Turbo-ov` (уже скачана);
  сервер через `openvino-env` на **запасном порту** (пользователь держит сервер
  на :8000); curl с `num_inference_steps=8, guidance_scale=0.0, size 512x512`;
  декод `b64_json` → валидный PNG; проверка `url` → data-URL.

## Вне скоупа

- Image2Image/Inpainting (`POST /v1/images/edits`) — отдельные пайплайны.
- Реальный хостинг файлов под `url` (отдаём data-URL).
- Лимиты/кэпы `n`, размера изображений, длины промпта.
- LoRA-адаптеры, прогресс-стриминг генерации, авто-дефолты под модель.
