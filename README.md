# ovinfer

OpenAI-compatible inference server and Python library on top of OpenVINO
(openvino-genai).

Two ways to use it:

1. **Server**: install `ovinfer[server]`, run `ovinfer server`, point any
   OpenAI client at it.
2. **Library**: add `ovinfer` to your project and call pipelines directly —
   `from ovinfer import Backend`.

Work in progress. Design docs live in `docs/superpowers/specs/`.
