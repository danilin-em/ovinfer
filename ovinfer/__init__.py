"""ovinfer: OpenVINO GenAI inference server with an OpenAI-compatible API.

v0.1 ships the HTTP server only; there is no public library API yet. The
internal core/transport boundary lives between `ovinfer.backend` (sync, no
FastAPI imports) and `ovinfer.server` (FastAPI transport).
"""

__version__ = "0.1.0"
