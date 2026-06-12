# InpaintingPipeline → POST /v1/images/edits (mask) — дизайн

Дата: 2026-06-12. Статус: одобрен, к реализации.

Реализация `og.InpaintingPipeline` в ovinfer: заглушка `501
InpaintingPipeline` на ветке с непустым `mask` в `backend.image_edit` (`POST
/v1/images/edits`) заменяется рабочим инференсом. Контракт эндпоинта — из спеки
2026-06-06-openai-server-design.md; маршрут и multipart-форма уже реализованы
для Image2Image (2026-06-11-image2image-design.md). Дизайн почти целиком
повторяет Image2Image — ниже выделены только отличия: третий тензор-аргумент
`mask_image`, гибридный декод маски и одно следствие для порядка проверок.

## Скоуп и переиспользование

Маршрут `/v1/images/edits` и его форма (`image`, `mask`, `prompt`, `n`, `size`,
`response_format` + диффузионные ext-поля, включая `strength`) **уже на месте**:
транспорт `images_edits` читает и пробрасывает `mask_bytes` без изменений.
Развилка `mask` уже живёт в `backend.image_edit`. Переиспользуются без
пересоздания: `ImageResult`, `_encode_pngs`, `_decode_image_tensor(...,
batched=True)`, `server._image_entry`, `server._parse_size`,
`registry.is_diffusion_model`, `errors.UnsupportedModel`/`InvalidImage`, все
четыре домен-ошибки в `_ERROR_MAP`. Эта итерация добавляет только
inpainting-специфичные части.

## Ключевые факты об og (2026.2.0, проверено)

- `og.InpaintingPipeline(models_path)` и `(models_path, device, **props)` —
  конструктор как у Image2Image.
- `generate(prompt: str, image: ov.Tensor, mask_image: ov.Tensor, **kwargs) ->
  ov.Tensor` — **третий позиционный аргумент `mask_image`** (отличие от
  Image2Image). kwargs те же, что у Image2Image:
  `num_images_per_prompt`, `guidance_scale`, `height`, `width`,
  `num_inference_steps`, `negative_prompt`, `rng_seed`, `generator`,
  **`strength`**, `prompt_2/3`, `negative_prompt_2/3`, `max_sequence_length`,
  `adapters`. Результат — uint8 RGB батч `(N, H, W, 3)`, как у Image2Image.
- Inpainting, как и Image2Image, **требует `vae_encoder`** (кодирует входные
  пиксели в латент). У локальной `OpenVINO/LCM_Dreamshaper_v7-int8-ov` он есть,
  есть и фабрика `og.InpaintingPipeline.latent_consistency_model` — LCM-inpainting
  поддержан. Маркер диффузии прежний (`model_index.json`, общий для
  Text2Image/Image2Image/Inpainting); модель с маркером, но несобираемая под
  inpainting падает на build и отдаёт 500 (как Z-Image у Text2Image; чистые
  build-ошибки — общий follow-up для всех пайплайнов).

## Маршрут и диспетчеризация

- `/v1/images/edits` — **multipart/form-data**, реализован для Image2Image.
- Намерение «edit» зафиксировано маршрутом; kind выбирается внутри
  `backend.image_edit` по наличию `mask`: непустой `mask` → Inpainting, иначе →
  Image2Image. Пустую (0-байт) загрузку `mask` трактуем как «mask нет».

## Семантика и декод маски (гибрид)

Единственный действительно новый код. `_decode_image_tensor` не годится для
маски: он делает `convert("RGB")` и теряет alpha-канал.

Две конвенции маски сталкиваются:

- **OpenAI** (`images.edit`): маска — RGBA-PNG, **прозрачные** пиксели (alpha=0)
  = «перерисовать», непрозрачные = «сохранить»; RGB-содержимое игнорируется.
- **diffusers/og** (`mask_image`): grayscale, **белое (255)** = «перерисовать»,
  чёрное (0) = «сохранить».

Связь: `og_mask = 255 − alpha`. **Выбран гибридный подход** (одобрен
пользователем): если у загруженной маски есть alpha-канал — OpenAI-семантика
(инвертируем alpha); если alpha нет — трактуем яркость как og-маску
(белое=править). Покрывает и реальные OpenAI-SDK-клиенты (шлют alpha), и простые
ч/б-маски, без footgun'а «маска без alpha молча ничего не правит» (риск строгого
OpenAI-режима).

Маска приводится к 4D NHWC `(1, H, W, 3)` uint8 — той же форме, что вход (og
inpainting-сэмпл грузит маску тем же 3-канальным батч-тензором, что и
изображение; Image2Image уже требует `batched=True` для входа). Одноканальная
«карта правки» броадкастится в 3 канала.

```python
def _decode_mask_tensor(raw: bytes) -> Any:
    """Decode an inpainting mask into an og mask_image tensor: 4D NHWC (1,H,W,3)
    uint8 where white (255) marks the region to repaint, black (0) the region to
    keep (og/diffusers convention).

    Hybrid input handling: an OpenAI-style RGBA mask marks the edit region with
    *transparent* pixels, so when an alpha channel is present it is used and
    inverted (alpha 0 -> 255 white -> repaint). A mask without alpha is read as
    plain luminance (white -> repaint), matching diffusers tutorials and simple
    b/w masks. The single-channel edit map is broadcast to 3 channels so the
    mask tensor matches the image tensor layout og's preprocessing expects.
    """
    import io

    import numpy as np
    import openvino as ov
    from PIL import Image, UnidentifiedImageError

    try:
        pic = Image.open(io.BytesIO(raw))
        pic.load()  # force decode now: broken data raises here
    except (UnidentifiedImageError, OSError) as exc:
        raise InvalidImage(f"could not decode mask: {exc}") from exc
    if pic.mode in ("RGBA", "LA") or "transparency" in pic.info:
        alpha = np.array(pic.convert("RGBA"))[..., 3]
        edit = 255 - alpha          # transparent (alpha 0) -> white -> repaint
    else:
        edit = np.array(pic.convert("L"))  # white -> repaint
    rgb = np.repeat(edit[..., np.newaxis], 3, axis=2).astype(np.uint8)
    return ov.Tensor(rgb[np.newaxis, ...])
```

Точная форма/число каналов маски, которую ждёт og, подтверждается на приёмке —
как Image2Image обнаружил необходимость `batched=True`. Плановая форма —
3-канальный батч NHWC по образцу og inpainting-сэмпла.

## Ядро (backend.py)

`backend.image_edit` — **сигнатура не меняется** (`mask: bytes | None` уже
есть). Меняется тело: ветка `mask` строит inpainting вместо 501, а **проверка
`is_diffusion_model` переезжает выше развилки `mask`** — inpainting тоже требует
диффузионную модель, поэтому non-diffusion → 400 для обеих веток. kwargs
строятся один раз (идентичны для img2img и inpainting; inpainting принимает те
же «ручки», включая `strength`).

Новый порядок проверок (стоимость растёт сверху вниз):

1. резолв модели → 404 `ModelNotFound`
2. пустой (0-байт) `mask` → `None`
3. `is_diffusion_model` → 400 `UnsupportedModel` *(теперь покрывает обе ветки)*
4. `_decode_image_tensor(image, batched=True)` → 400 `InvalidImage`
5. собрать только заданные kwargs (`None`-поля не передаём; `0.0` проходит;
   `num_images_per_prompt=n` всегда)
6. **если `mask is not None`:** `_decode_mask_tensor(mask)` → 400 `InvalidImage`
   → `_pipeline(model_id, "inpainting")` →
   `pipe.generate(prompt, image_tensor, mask_tensor, **kwargs)`
   **иначе:** `_pipeline(model_id, "image2image")` →
   `pipe.generate(prompt, image_tensor, **kwargs)`
7. тензор `(N,H,W,3)` → `_encode_pngs` → `ImageResult`

Порядок: модель валидируется до картинок; декоды (дёшево) — до постройки
пайплайна (дорого), чтобы битый вход/маска давали 400 без загрузки модели.
Генерация — один блокирующий вызов; стриминга/отмены по токенам нет (как у
Text2Image/Image2Image).

Ветка в `_create_pipeline`: `kind == "inpainting"` →
`og.InpaintingPipeline(model_dir, device)`.

*Отвергнутая альтернатива:* конструктор-overload
`og.InpaintingPipeline(image2image_pipe)` для переиспользования уже
скомпилированного img2img-пайплайна. Экономит компиляцию, но вводит кросс-kind
зависимость в LRU-кэше — YAGNI; отдельные kind проще и симметричнее.

## Транспорт (server.py)

Роут `images_edits` — **без изменений** (уже читает `mask.file.read()` и
пробрасывает `mask_bytes` в `backend.image_edit`). Меняется только строка карты
модуля в docstring:
`InpaintingPipeline -> POST /v1/images/edits (mask) (implemented)`.

## Конфиг / preload / NPU

- `config.PIPELINE_KINDS` += `"inpainting"` — `--preload org/model:inpainting`
  становится валидным; отдельный кэш-kind от `image2image`.
- `registry.is_diffusion_model` — без изменений (маркер `model_index.json`
  общий для Text2Image/Image2Image/Inpainting).
- `npu.py` — без изменений: `reject_reason` для `"inpainting"` проваливается в
  `None` (падение на NPU не проверялось, как и у VLM/Text2Image/Image2Image —
  не флагуем).

## Ошибки

Порядок и коды (`errors.py` и `_ERROR_MAP` — без изменений, все домен-ошибки уже
замаплены):

- Несуществующая модель → 404 `model_not_found` (резолв первым).
- Не-диффузионная модель → 400 `invalid_request_error`/`unsupported_model`
  (`UnsupportedModel`) — **теперь и для маски тоже** (см. ниже).
- Битая/нераспознанная входная картинка → 400 `invalid_request_error`/
  `invalid_image` (`InvalidImage`, из `_decode_image_tensor`).
- Битая/нераспознанная маска → 400 `invalid_request_error`/`invalid_image`
  (`InvalidImage`, из `_decode_mask_tensor`).
- Кривой `size` → 400 `invalid_request_error`/`invalid_size` (транспорт).
- `n < 1` → 422 (`Form(..., ge=1)`).

## Следствие: меняется запиненное поведение (коммит `9603b32`)

Реализация inpainting **отменяет** порядок «mask побеждает non-diffusion (501
над 400)»: непустой `mask` больше не упирается в 501 до диффузионной проверки.
Теперь `is_diffusion_model` стоит выше развилки `mask`, поэтому non-diffusion
модель → 400 независимо от наличия маски. Три теста в
`tests/test_image2image.py` обновляются:

- `test_non_empty_mask_is_501_inpainting_without_building_pipeline` —
  **удалить** (mask больше не 501; поведение переезжает в `test_inpainting.py`).
- `test_mask_beats_non_diffusion_model` — **изменить**: non-diffusion + mask
  теперь → 400 `UnsupportedModel`. Переименовать в
  `test_non_diffusion_model_is_unsupported_even_with_mask`.
- транспортный `test_non_empty_mask_returns_501` — **удалить** (mask теперь даёт
  200-конверт; 501-маппинг и так покрыт audio/video-заглушками).
- комментарий в `test_unknown_model_404s_before_anything_else` («model
  resolution must win the 404 over the mask 501») — поправить (больше не «501»).

## Граничные случаи

- `mask` пустой (0 байт) → «нет mask» → Image2Image (как сейчас).
- `mask` без alpha (ч/б PNG, palette-`transparency`) → яркость, белое=править.
- `mask` с alpha (OpenAI-SDK) → alpha инвертируется (прозрачное=править).
- Размеры `mask` ≠ `image` → ответственность клиента (OpenAI тоже требует
  совпадения); рассинхрон → ошибка og → 500. Без авто-resize (YAGNI).
- `strength`/`guidance_scale = 0.0` → проходит (значащее значение).
- `seed` опущен → дефолтный `rng_seed=42`; задан → вариативно.
- `n > 1` → тензор `(n,H,W,3)`, `data` из n элементов.
- `response_format="url"` → data-URL; `"b64_json"` (дефолт) → base64.

## Тестирование

Новый `tests/test_inpainting.py` (зеркалит `tests/test_image2image.py`; `og`
не импортируется — чистые хелперы тестируются напрямую, dispatch стабит
`_pipeline`, транспорт стабит весь backend):

- **Юнит**: `"inpainting"` в `PIPELINE_KINDS` и `parse_preload`;
  `_decode_mask_tensor` — RGBA-маска: прозрачная зона → белое (255) в тензоре,
  непрозрачная → чёрное (инверсия alpha); L-маска (без alpha): белое остаётся
  белым (править); `palette`+`transparency` обрабатывается как alpha; форма
  результата `(1,H,W,3)`; мусор → `InvalidImage`.
- **Дисптач** (стаб `_pipeline`, реальный Backend): непустой mask → запрашивается
  kind `"inpainting"`; `prompt` + image-тензор + mask-тензор (третий поз. арг.) и
  только заданные kwargs доходят до `generate` (`n`→`num_images_per_prompt`,
  `seed`→`rng_seed`, `strength`→`strength`, `0.0` проходит, опущенные —
  отсутствуют); non-diffusion + mask → 400 `UnsupportedModel` без пайплайна;
  битая маска → 400 `InvalidImage` без пайплайна; битый image (+mask) → 400 без
  пайплайна; несуществующая модель (+mask) → 404 до всего.
- **Контракт** (TestClient + multipart + стаб-Backend): mask-загрузка → 200,
  форма ответа `b64_json`; mask-байты доходят до backend.
- **Правки `test_image2image.py`**: см. раздел «Следствие» (удалить 2 теста,
  изменить 1, поправить комментарий).

**Приёмка на реальной модели**: `OpenVINO/LCM_Dreamshaper_v7-int8-ov` (есть
`vae_encoder`); сервер через `openvino-env` на **запасном порту** (пользователь
держит сервер на :8000); проверки: img2img без mask остаётся 200/PNG; запрос с
RGBA-маской (прозрачный круг по центру) → 200/PNG; ч/б-маска → 200/PNG;
рассинхрон размеров mask/image и мусор-маска — диагностические проверки
(ожидаем 500 / 400 соответственно).

## Вне скоупа

- Авто-resize маски под размер входа (клиент шлёт совпадающие размеры, как
  требует OpenAI).
- Дедицированные inpainting-чекпойнты (9-канальный UNet) сверх того, что og
  делает сам для обычных SD/LCM-моделей.
- Реальный хостинг файлов под `url` (отдаём data-URL).
- Лимиты/кэпы `n`, размера изображений, длины промпта.
- LoRA-адаптеры, прогресс-стриминг генерации, множественные входные картинки.
- Чистые build-ошибки несобираемой диффузионной модели → 500 (общий follow-up
  для всех пайплайнов).
