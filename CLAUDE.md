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

## Состояние (2026-06-11, обновлять при смене фазы)

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
- Спека v0.1 (одобрена и реализована): `docs/superpowers/specs/2026-06-07-ovinfer-v0.1-design.md`.
- Спека контракта (маппинг 11 og.*-пайплайнов на эндпоинты, схемы, ошибки, SSE):
  `docs/superpowers/specs/2026-06-06-openai-server-design.md`.
- Нюанс этой машины: реальное дерево моделей —
  `~/.openvino/models/huggingface.co/<org>/<model>` (три уровня), поэтому
  `OVINFER_MODELS_ROOT=~/.openvino/models/huggingface.co`.
- Следующие шаги: follow-up в `~/.openvino` (см. спеку, раздел «Связь с
  ~/.openvino»); спека каталога моделей v0.2.

## Язык

- Код, комментарии, докстринги, README, коммиты — **только на английском**.
- Русский — только в общении с пользователем: отчёты, ответы, объяснения в чате.
