"""Tests for the rerank feature: Backend.rerank (core) and POST /v1/rerank
(transport). og is never imported: the core tests stub out _pipeline, the
transport tests stub out the whole backend.
"""

import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from ovinfer.backend import Backend, RerankResult
from ovinfer.errors import ModelNotFound

MODEL = "org/reranker"


def _models_root() -> Path:
    """Temp models_root with one model dir (an *.xml makes it a model)."""
    root = Path(tempfile.mkdtemp())
    model_dir = root / MODEL
    model_dir.mkdir(parents=True)
    (model_dir / "openvino_model.xml").write_text("<net/>")
    return root


class _FakePipe:
    """Stands in for og.TextRerankPipeline: returns canned (index, score)."""

    def __init__(self, raw: list[tuple[int, float]]) -> None:
        self.raw = raw
        self.calls: list[tuple[str, list[str]]] = []

    def rerank(self, query: str, texts: list[str]) -> list[tuple[int, float]]:
        self.calls.append((query, list(texts)))
        return self.raw


class _StubPipelineBackend(Backend):
    """Backend with _pipeline stubbed out; records (model_id, kind) requests."""

    def __init__(self, models_root: Path, pipe: _FakePipe) -> None:
        super().__init__(models_root=models_root, default_device="CPU")
        self._pipe = pipe
        self.pipeline_requests: list[tuple[str, str]] = []

    @contextmanager
    def _pipeline(self, model_id: str, kind: str):
        self.pipeline_requests.append((model_id, kind))
        yield self._pipe


class BackendRerankTest(unittest.TestCase):
    def setUp(self):
        self.root = _models_root()
        self.pipe = _FakePipe([(2, 0.97), (0, 0.12), (1, 0.03)])
        self.backend = _StubPipelineBackend(self.root, self.pipe)
        self.texts = ["doc a", "doc b", "doc c"]

    def test_returns_all_results_in_og_order_without_top_n(self):
        result = self.backend.rerank(MODEL, "q", self.texts)
        self.assertEqual(result, RerankResult(results=[(2, 0.97), (0, 0.12), (1, 0.03)]))

    def test_requests_rerank_pipeline_kind(self):
        self.backend.rerank(MODEL, "q", self.texts)
        self.assertEqual(self.backend.pipeline_requests, [(MODEL, "rerank")])

    def test_passes_query_and_texts_to_pipeline(self):
        self.backend.rerank(MODEL, "the query", self.texts)
        self.assertEqual(self.pipe.calls, [("the query", self.texts)])

    def test_top_n_slices_results(self):
        result = self.backend.rerank(MODEL, "q", self.texts, top_n=1)
        self.assertEqual(result.results, [(2, 0.97)])

    def test_top_n_larger_than_documents_is_harmless(self):
        result = self.backend.rerank(MODEL, "q", self.texts, top_n=10)
        self.assertEqual(len(result.results), 3)

    def test_empty_texts_short_circuit_without_pipeline(self):
        result = self.backend.rerank(MODEL, "q", [])
        self.assertEqual(result, RerankResult(results=[]))
        self.assertEqual(self.backend.pipeline_requests, [])

    def test_empty_texts_still_resolve_the_model(self):
        with self.assertRaises(ModelNotFound):
            self.backend.rerank("org/ghost", "q", [])


if __name__ == "__main__":
    unittest.main()
