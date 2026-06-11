# Image2ImagePipeline → POST /v1/images/edits — дизайн

Дата: 2026-06-11. Статус: одобрен, к реализации.

Реализация `og.Image2ImagePipeline` в ovinfer: заглушка 501 на ветке без mask в
`backend.image_edit` (`POST /v1/images/edits`) заменяется рабочим инференсом.
Контракт эндпоинта — из спеки 2026-06-06-openai-server-design.md; детали маппинга
параметров, загрузки/декода входной картинки и диспетчеризации фиксируются здесь.
Дизайн почти целиком повторяет Text2Image (2026-06-11-text2image-design.md) —
ниже выделены только отличия.

## Ключевые факты об og (2026.2.0)

- `og.Image2ImagePipeline(models_path)` и `(models_path, device, **props)`;
  `generate(prompt: str, image: ov.Tensor, **kwargs) -> ov.Tensor`. Вход `image` —
  `ov.Tensor` uint8 RGB HWC (ровно то, что `_load_image_tensor` строит для VLM).
  Результат — uint8 RGB батч `(N, H, W, 3)`, как у Text2Image.
- Параметры `generate` (kwargs) — те же, что у Text2Image, плюс определяющий
  img2img knob **`strength`**: `num_images_per_prompt`, `guidance_scale`, `height`,
  `width`, `num_inference_steps`, `negative_prompt`, `rng_seed`, `generator`,
  `strength`, `prompt_2/3`, `negative_prompt_2/3`, `max_sequence_length`, `adapters`.
- **`strength`** управляет балансом «сохранить вход ↔ следовать промпту»: 0.0 ≈
  вернуть вход, 1.0 = полностью зашумить (фактически text2image). Дефолт og — 0.8
  (конвенция diffusers).
- **Image2Image требует `vae_encoder`** (пиксели входа → латент, затем частичное
  зашумление по `strength`), в отличие от Text2Image, которому нужен только
  `vae_decoder`. Локальная `OpenVINO/LCM_Dreamshaper_v7-int8-ov` содержит
  `vae_encoder` — приёмка на ней проходит. Маркер диффузии прежний
  (`model_index.json`); модель с маркером, но без `vae_encoder`/неподдерживаемой
  архитектурой падает на build и отдаёт 500 (как Z-Image у Text2Image).

## Маршрут и диспетчеризация

- `/v1/images/edits` — **multipart/form-data** (не JSON, в отличие от
  `/v1/images/generations`): `image` приходит файлом-загрузкой (`UploadFile`).
- Намерение «edit» зафиксировано маршрутом, поэтому диспетчеризация по форме
  запроса (как VLM-vs-LLM в chat) не нужна: kind жёстко `"image2image"`.
- Развилка mask остаётся в `backend.image_edit`: **mask непуст → 501
  `PipelineNotImplemented("InpaintingPipeline")`** (отдельный пайплайн, отдельная
  итерация); mask нет → Image2Image. Пустую (0-байт) загрузку mask трактуем как
  «mask нет» — иначе можно молча выбрать inpainting с пустым mask-тензором (уже
  так в текущем стабе).

## Маппинг параметров (выбранный подход)

`/v1/images/edits` стандартного OpenAI несёт `image/prompt/mask/n/size/
response_format` — для diffusion-img2img этого мало. Схема дополняется
**опциональными Form-полями-расширениями** (на multipart нельзя положиться на
`extra="allow"`, как в JSON-схеме generations — поля объявляются явно), которые
pass-through уходят в `og`; при отсутствии — дефолты `og`:

| Form-поле | og kwarg | Примечание |
|---|---|---|
| `image` (файл) | `image` (поз. арг.) | обязательно; декод в `ov.Tensor` |
| `prompt` | `prompt` | обязательно |
| `n` (=1, `ge=1`) | `num_images_per_prompt` | передаётся всегда |
| `size` ("ШxВ") | `width`, `height` | `None`/`"auto"` → не передаём (og берёт из входа) |
| `strength?` | `strength` | ext.: баланс сохранения входа; дефолт og 0.8 |
| `num_inference_steps?` | `num_inference_steps` | ext. |
| `guidance_scale?` | `guidance_scale` | ext. |
| `negative_prompt?` | `negative_prompt` | ext. |
| `seed?` | `rng_seed` | ext.: вариативность поверх дефолтного 42 |
| `mask?` (файл) | — | непустой → 501 Inpainting (см. выше) |
| `response_format` | — | wire-формат, см. ниже |

В `og.generate` передаются **только заданные** kwargs: `None`-поля не передаются,
чтобы не затирать дефолты пайплайна. `0.0` — значащее значение и проходит (важно
для `guidance_scale`/`strength`). `num_images_per_prompt=n` передаётся всегда.

Отвергнутые альтернативы: собственный дефолт `strength` (всегда уходит в og) —
лишняя неявная логика, расхождение с дефолтом og и с правилом «не клобберить
дефолты пайплайна», YAGNI. Inpainting в этой итерации — отдельный пайплайн с
mask-семантикой OpenAI (прозрачные зоны = редактировать), требует доп.
проработки; остаётся 501.

## Ядро (backend.py)

- **Рефактор переиспользования.** Из `_load_image_tensor(url)` выделяется хелпер
  `_decode_image_tensor(raw: bytes) -> ov.Tensor` (PIL-декод → `convert("RGB")` →
  `ov.Tensor(np.array(...))`; `InvalidImage` на битых/нераспознанных байтах).
  `_load_image_tensor` становится `_decode_image_tensor(_fetch_image_bytes(url))`.
  img2image кормит декодер байтами загрузки напрямую — тем же кодом, которому уже
  доверяет VLM-путь.
- Ветка в `_create_pipeline`: `kind == "image2image"` →
  `og.Image2ImagePipeline(model_dir, device)`.
- `image_edit` (был стаб 501) переписывается под форму, параллельную
  `image_generate`, и возвращает `ImageResult`:

```python
def image_edit(self, model_id, prompt, image: bytes, mask: bytes | None, *,
               n=1, width=None, height=None, num_inference_steps=None,
               guidance_scale=None, negative_prompt=None, strength=None,
               seed=None) -> ImageResult
```

Поток: резолв модели (404) → mask непуст → **501 Inpainting** →
`is_diffusion_model` (400 `UnsupportedModel`) → `_decode_image_tensor(image)`
(400 `InvalidImage`) → собрать только заданные kwargs →
`_pipeline(model_id, "image2image")` → `pipe.generate(prompt, image_tensor,
**kwargs)` → тензор `(N,H,W,3)` → `_encode_pngs` (переиспользуется).

Порядок проверок: модель валидируется до входной картинки; декод (дёшево) — до
постройки пайплайна (дорого), чтобы битая загрузка давала 400 без загрузки модели
(зеркалит eager-загрузку изображений в VLM). Генерация — один блокирующий вызов;
стриминга/отмены по токенам нет (роут `def` исполняется в threadpool Starlette,
`/health` отзывчив), как у Text2Image.

## Транспорт (server.py)

Роут `images_edits` переписывается (Form-поля + парс size + конверт):

```python
@app.post("/v1/images/edits")
def images_edits(
    model: str = Form(...),
    prompt: str = Form(...),
    image: UploadFile = File(...),
    mask: UploadFile | None = File(None),
    n: int = Form(1, ge=1),
    size: str | None = Form(None),
    response_format: str = Form("b64_json"),
    num_inference_steps: int | None = Form(None),
    guidance_scale: float | None = Form(None),
    negative_prompt: str | None = Form(None),
    strength: float | None = Form(None),
    seed: int | None = Form(None),
) -> Any
```

- `size` парсится существующим `_parse_size` (кривой формат → 400
  `invalid_request_error`/`invalid_size`, JSONResponse в роуте — как в generations).
- Читает байты `image`/`mask` (`mask` — `None`, если файла нет).
- Вызывает `backend.image_edit(...)` и строит конверт через существующий
  `_image_entry`:

```json
{"created": <ts>, "data": [{"b64_json": "..."}]}
```

Кодирование элемента (переиспользуется `_image_entry`): `b64_json` → base64(PNG);
`url` → `"data:image/png;base64," + base64(PNG)`. Дефолт `response_format` —
`b64_json` (как в текущем стабе и в generations).

Карта модуля в docstring обновляется:
`Image2ImagePipeline -> POST /v1/images/edits (implemented)`;
строка `InpaintingPipeline ... (501)` остаётся.

## Ошибки

Порядок и коды:

- Несуществующая модель → 404 `model_not_found` (резолв первым, до пайплайна).
- Непустой `mask` → 501 `not_implemented`/`pipeline_not_implemented`
  (`PipelineNotImplemented("InpaintingPipeline")`).
- Не-диффузионная модель → 400 `invalid_request_error`/`unsupported_model`
  (`UnsupportedModel`).
- Битая/нераспознанная входная картинка → 400 `invalid_request_error`/
  `invalid_image` (`InvalidImage`, из `_decode_image_tensor`).
- Кривой `size` → 400 `invalid_request_error`/`invalid_size` в транспорте.
- `n < 1` → 422 (Pydantic/`Form(..., ge=1)`).

`errors.py` и `_ERROR_MAP` — без изменений: все четыре домен-ошибки уже
замаплены (`ModelNotFound`/`UnsupportedModel`/`InvalidImage`/
`PipelineNotImplemented`).

## Конфиг / preload / NPU

- `config.PIPELINE_KINDS` += `"image2image"` — `--preload org/model:image2image`
  становится валидным.
- `registry.is_diffusion_model` — без изменений (маркер `model_index.json` общий
  для Text2Image/Image2Image/Inpainting).
- `npu.reject_reason` для `"image2image"` → `None` (падение на NPU не
  проверялось, как и VLM/Text2Image — не флагуем). Изменений в `npu.py` не
  требуется (kind проваливается в `None`).

## Граничные случаи

- `mask` пустой (0 байт) → трактуется как «нет mask» → Image2Image.
- `size` опущен → `width/height` не передаются, og берёт размер из входа.
- `n > 1` → тензор `(n,H,W,3)`, `data` из n элементов.
- `strength`/`guidance_scale = 0.0` → проходит (значащее значение, не отбрасывается).
- `seed` опущен → дефолтный `rng_seed=42` (детерминированно); задан → вариативно.
- `response_format="url"` → data-URL; `"b64_json"` (дефолт) → base64.

## Тестирование

- **Юнит**: `_decode_image_tensor` (валидный PNG-байт-стрим → `ov.Tensor` нужной
  формы; мусор → `InvalidImage`); `image2image` в `PIPELINE_KINDS` и
  `parse_preload`.
- **Дисптач** (стаб `_pipeline`, реальный Backend): запрашивается kind
  `"image2image"`; `prompt` + image-тензор и только заданные kwargs доходят до
  `generate` (`n`→`num_images_per_prompt`, `seed`→`rng_seed`, `strength`→
  `strength`, опущенные — отсутствуют; `0.0` проходит); непустой mask → 501 без
  построения пайплайна; не-диффузионная модель → 400 `UnsupportedModel` без
  пайплайна; несуществующая модель → 404 до пайплайна.
- **Контракт** (TestClient + multipart + стаб-Backend): форма ответа `b64_json`;
  `url` → data-URL; проброс ext-полей в backend (вкл. `strength`); кривой `size`
  → 400; непустой mask → 501; не-диффузионная модель → 400 `unsupported_model`;
  несуществующая → 404; `n<1` → 422.
- **Приёмка на реальной модели**: `OpenVINO/LCM_Dreamshaper_v7-int8-ov` (есть
  `vae_encoder`, уже скачана); сервер через `openvino-env` на **запасном порту**
  (пользователь держит сервер на :8000); `curl -F model=... -F prompt=... -F
  image=@<png> -F num_inference_steps=4 -F strength=0.6`; декод `b64_json` →
  валидный PNG; проверка `url` → data-URL; проверка непустого mask → 501.

## Вне скоупа

- Inpainting (`og.InpaintingPipeline`, mask-ветка) — отдельный пайплайн с
  mask-семантикой OpenAI; остаётся 501.
- Реальный хостинг файлов под `url` (отдаём data-URL).
- Лимиты/кэпы `n`, размера изображений, длины промпта.
- LoRA-адаптеры, прогресс-стриминг генерации, авто-дефолты под модель,
  множественные входные картинки (OpenAI `image[]`).
