"""Tests for embedding pooling selection: registry.embedding_pooling.

og is never imported. The pooling decision is pure model-dir introspection
(config.json architecture + sentence-transformers 1_Pooling/config.json), so
it is tested against temp directories holding only those marker files.
"""

import json
import tempfile
import unittest
from pathlib import Path

from ovinfer import registry


def _model_dir(*, architecture: str | None = None, pooling: dict | None = None) -> Path:
    """Temp model dir with an optional config.json arch and 1_Pooling config."""
    root = Path(tempfile.mkdtemp())
    if architecture is not None:
        (root / "config.json").write_text(json.dumps({"architectures": [architecture]}))
    if pooling is not None:
        pdir = root / "1_Pooling"
        pdir.mkdir()
        (pdir / "config.json").write_text(json.dumps(pooling))
    return root


class EmbeddingPoolingTest(unittest.TestCase):
    def test_causal_lm_uses_last_token(self):
        # A decoder LM (causal attention): the first token sees only itself, so
        # CLS pooling collapses unrelated texts together. Last-token is correct.
        d = _model_dir(architecture="Qwen3ForCausalLM")
        self.assertEqual(registry.embedding_pooling(d), "last_token")

    def test_encoder_uses_cls(self):
        # A BERT-style encoder ships a real [CLS] token: keep og's default.
        d = _model_dir(architecture="BertModel")
        self.assertEqual(registry.embedding_pooling(d), "cls")

    def test_missing_config_defaults_to_cls(self):
        d = _model_dir()
        self.assertEqual(registry.embedding_pooling(d), "cls")

    def test_pooling_config_mean_overrides_architecture(self):
        # Sentence-transformers metadata is authoritative when present: a mean
        # pooling marker wins even over a causal-LM architecture.
        d = _model_dir(
            architecture="Qwen3ForCausalLM",
            pooling={"pooling_mode_mean_tokens": True},
        )
        self.assertEqual(registry.embedding_pooling(d), "mean")

    def test_pooling_config_lasttoken(self):
        d = _model_dir(pooling={"pooling_mode_lasttoken": True})
        self.assertEqual(registry.embedding_pooling(d), "last_token")

    def test_pooling_config_cls(self):
        d = _model_dir(
            architecture="MistralModel",
            pooling={"pooling_mode_cls_token": True},
        )
        self.assertEqual(registry.embedding_pooling(d), "cls")


if __name__ == "__main__":
    unittest.main()
