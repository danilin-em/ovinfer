# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Что это за проект

ovinfer — самодостаточная платформа инференса на OpenVINO (openvino-genai):
OpenAI-совместимый HTTP-сервер (`ovinfer server`, console-script) — любой
OpenAI-клиент ходит по HTTP. Внутри пакет разделён на sync-ядро (`Backend`,
без FastAPI-импортов) и тонкий транспорт; публичный библиотечный API
(`uv add ovinfer`) отложен до реального потребителя — граница ядро/транспорт
позволит открыть его дёшево.

Проект выделен из `~/.openvino` (машинная обвязка vendored Intel-стека) — это **два
разных проекта**; ovinfer про обвязку не знает. На этой машине любой запуск
OpenVINO-кода — через `openvino-env <команда>` (см. глобальные правила пользователя).

## Состояние (2026-06-12, обновлять при смене фазы)

- **v0.1 реализован и принят 2026-06-07**: пакет `ovinfer/` (sync-ядро + транспорт),
  LLM chat со стримингом и embeddings работают на реальных моделях; приёмка по
  чек-листу спеки пройдена. План: `docs/superpowers/plans/2026-06-07-ovinfer-v0.1.md`.
- **Rerank реализован 2026-06-10**: `POST /v1/rerank` через og.TextRerankPipeline
  (Jina-формат ответа); спека: `docs/superpowers/specs/2026-06-10-rerank-design.md`.
- **VLM реализован 2026-06-11**: `og.VLMPipeline` на `POST /v1/chat/completions`
  — модель-VLM (есть `openvino_vision_embeddings_model.xml`) обслуживает и
  text-only, и image-запросы; диспетчеризация по файлам модели; изображения из
  data:/http(s) URL (приёмка на `gemma-3-4b-it-int4-cw-ov`). Спека:
  `docs/superpowers/specs/2026-06-10-vlm-design.md`; план:
  `docs/superpowers/plans/2026-06-11-vlm.md`.
- **Text2Image реализован и принят 2026-06-11**: `og.Text2ImagePipeline` на
  `POST /v1/images/generations` — диффузионная модель (есть `model_index.json`)
  отдаёт PNG (`b64_json`/data-URL); диффузионные «ручки» (`num_inference_steps`,
  `guidance_scale`, `negative_prompt`, `seed`) как опциональные поля-расширения.
  Приёмка end-to-end на `OpenVINO/LCM_Dreamshaper_v7-int8-ov` (4 шага, 512×512,
  CPU). **Важно: `og.Text2ImagePipeline` поддерживает только SD/SDXL/SD3/Flux/LCM**
  — у локальной `hsuwill000/Z-Image-Turbo-ov` `_class_name: ZImagePipeline`, og
  её не знает (нужен `optimum-intel`/`OVZImagePipeline`, другой рантайм). Модель с
  `model_index.json`, но неподдерживаемой og архитектурой даёт 500 (как любая
  несобираемая модель; чистые build-ошибки — follow-up для всех пайплайнов).
  Спека: `docs/superpowers/specs/2026-06-11-text2image-design.md`; план:
  `docs/superpowers/plans/2026-06-11-text2image.md`.
- **Image2Image реализован и принят 2026-06-12**: `og.Image2ImagePipeline` на
  `POST /v1/images/edits` (ветка без mask) — multipart-загрузка `image` +
  `prompt` отдаёт PNG (`b64_json`/data-URL); диффузионные «ручки»
  (`num_inference_steps`, `guidance_scale`, `negative_prompt`, `seed`) плюс
  img2img-knob `strength` как опциональные Form-поля-расширения. Декод входа —
  общий с VLM хелпер `_decode_image_tensor`, но **img2img требует 4D NHWC
  `(1,H,W,3)`** (`batched=True`): у `og.Image2ImagePipeline` вход модели `[?,?,?,?]`
  и голый 3D HWC отвергается (VLM-путь остаётся 3D). Непустой `mask` → Inpainting (реализован, см. ниже); не-диффузионная модель → 400. Требует
  `vae_encoder` (пиксели входа → латент), в отличие от Text2Image. Приёмка
  end-to-end на `OpenVINO/LCM_Dreamshaper_v7-int8-ov` (4 шага, `strength=0.6`,
  512×512, CPU): img2img→200/PNG, `url`→data-URL, mask→501 (на тот момент; ныне
  Inpainting, см. ниже), non-diffusion→400.
  Спека: `docs/superpowers/specs/2026-06-11-image2image-design.md`; план:
  `docs/superpowers/plans/2026-06-12-image2image.md`.
- **Inpainting реализован 2026-06-12**: `og.InpaintingPipeline` на
  `POST /v1/images/edits` (ветка с непустым `mask`) — multipart-загрузка `image`
  + `mask` + `prompt` отдаёт PNG (`b64_json`/data-URL); те же диффузионные
  «ручки», что у Image2Image (включая `strength`). `generate(prompt, image,
  mask_image, **kwargs)` — маска третьим позиционным тензором. Декод маски —
  новый хелпер `_decode_mask_tensor` с **гибридной семантикой**: есть alpha
  (OpenAI-SDK) → прозрачное = править (инверсия alpha); нет alpha → яркость как
  og-маска (белое = править); форма `(1,H,W,3)`. Проверка `is_diffusion_model`
  переехала выше развилки `mask`, поэтому не-диффузионная модель → 400 для обеих
  веток edit (это отменяет прежний порядок «mask→501 над non-diffusion→400»).
  Требует `vae_encoder`, как Image2Image. Спека:
  `docs/superpowers/specs/2026-06-12-inpainting-design.md`; план:
  `docs/superpowers/plans/2026-06-12-inpainting.md`.
- Спека v0.1 (одобрена и реализована): `docs/superpowers/specs/2026-06-07-ovinfer-v0.1-design.md`.
- Спека контракта (маппинг 11 og.*-пайплайнов на эндпоинты, схемы, ошибки, SSE):
  `docs/superpowers/specs/2026-06-06-openai-server-design.md`.
- Нюанс этой машины: реальное дерево моделей —
  `~/.openvino/models/huggingface.co/<org>/<model>` (три уровня), поэтому
  `OVINFER_MODELS_ROOT=~/.openvino/models/huggingface.co`.
- **Whisper (ASR) спланирован 2026-06-12**: `og.WhisperPipeline` на `POST
  /v1/audio/transcriptions|translations` (batch) — заполняет готовые стабы
  `backend.transcribe`/`translate`. Декод аудио через PyAV (`av`) в моно 16 кГц
  float; пять `response_format` (json/text/verbose_json/srt/vtt); маркер модели
  `openvino_encoder_model.xml` (`is_whisper_model`), non-whisper → 400. Спека:
  `docs/superpowers/specs/2026-06-12-whisper-design.md` (план — следующим шагом).
- Следующие шаги: follow-up в `~/.openvino` (см. спеку, раздел «Связь с
  ~/.openvino»); спека каталога моделей v0.2; аудио-follow-up'ы — стриминг
  ответа (`stream:true`) и realtime-вход (WebSocket): планы в
  `docs/superpowers/follow-ups/audio-streaming-and-realtime.md`.

## Язык

- Код, комментарии, докстринги, README, коммиты — **только на английском**.
- Русский — только в общении с пользователем: отчёты, ответы, объяснения в чате.
