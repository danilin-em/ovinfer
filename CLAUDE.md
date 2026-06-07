# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Что это за проект

ovinfer — самодостаточная платформа инференса на OpenVINO (openvino-genai):
Python-библиотека + OpenAI-совместимый HTTP-сервер. Два пути использования:

1. **Сервер**: `ovinfer server` (console-script, extra `[server]`) — любой
   OpenAI-клиент ходит по HTTP.
2. **Библиотека**: `uv add ovinfer` в свой проект, `from ovinfer import Backend` —
   прямые вызовы без HTTP. Ядро синхронное, FastAPI в нём не импортируется.

Проект выделен из `~/.openvino` (машинная обвязка vendored Intel-стека) — это **два
разных проекта**; ovinfer про обвязку не знает. На этой машине любой запуск
OpenVINO-кода — через `openvino-env <команда>` (см. глобальные правила пользователя).

## Состояние (2026-06-07, обновлять при смене фазы)

- Контракт API финальный: `server.py` в корне — перенесённый из `~/.openvino` каркас,
  все инференс-эндпоинты отвечают 501; при реализации v0.1 он растворяется в пакет
  `ovinfer/` и удаляется.
- Спека v0.1 (структура пакета, sync-ядро, LRU-кеш пайплайнов, LLM chat со
  стримингом, CLI): `docs/superpowers/specs/2026-06-07-ovinfer-v0.1-design.md` —
  **ждёт ревью пользователя**; после одобрения — навык writing-plans, затем реализация.
- Спека контракта (маппинг 11 og.*-пайплайнов на эндпоинты, схемы, ошибки, SSE):
  `docs/superpowers/specs/2026-06-06-openai-server-design.md`.
- `pyproject.toml`, пакета и venv ещё нет — появятся по плану v0.1.

## Язык

- Код, комментарии, докстринги, README, коммиты — **только на английском**.
- Русский — только в общении с пользователем: отчёты, ответы, объяснения в чате.
