"""Domain errors. No HTTP semantics here: status mapping lives in the transport."""


class OvinferError(Exception):
    """Base class for ovinfer domain errors."""


class ModelNotFound(OvinferError):
    """The OpenAI `model` field does not resolve to a local model directory."""

    def __init__(self, model_id: str, models_root) -> None:
        super().__init__(f"Model '{model_id}' not found in {models_root}")
        self.model_id = model_id


class PipelineNotImplemented(OvinferError):
    """The endpoint exists in the contract but its pipeline is not wired yet."""

    def __init__(self, pipeline: str) -> None:
        super().__init__(f"{pipeline} backend is not implemented yet")
        self.pipeline = pipeline


class ConfigError(OvinferError):
    """Invalid configuration (bad device-map syntax, preload over capacity, ...)."""
