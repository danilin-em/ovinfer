"""Tests for the rerank feature: Backend.rerank (core) and POST /v1/rerank
(transport). og is never imported: the core tests stub out _pipeline, the
transport tests stub out the whole backend.
"""

import tempfile
import unittest
from contextlib import contextmanager
from pathlib import Path

from fastapi.testclient import TestClient

from ovinfer.backend import Backend, RerankResult
from ovinfer.errors import ModelNotFound
from ovinfer.server import create_app

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


class _StubBackend:
    """Duck-typed Backend for transport tests; returns canned RerankResult."""

    def __init__(self, results: list[tuple[int, float]] | Exception) -> None:
        self.results = results
        self.calls: list[tuple[str, str, list[str], int | None]] = []

    def rerank(self, model_id, query, texts, top_n=None) -> RerankResult:
        self.calls.append((model_id, query, texts, top_n))
        if isinstance(self.results, Exception):
            raise self.results
        return RerankResult(results=self.results)


def _client(results) -> tuple[TestClient, _StubBackend]:
    backend = _StubBackend(results)
    return TestClient(create_app(backend)), backend


_DOCS = ["Berlin is the capital of Germany.",
         "The cell powerhouse.",
         "Paris is the capital of France."]


class RerankRouteTest(unittest.TestCase):
    def _post(self, client, **overrides):
        body = {"model": MODEL, "query": "capital of France?", "documents": _DOCS}
        body.update(overrides)
        return client.post("/v1/rerank", json=body)

    def test_jina_response_format(self):
        client, _ = _client([(2, 0.97), (0, 0.12)])
        resp = self._post(client)
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json(), {
            "model": MODEL,
            "results": [
                {"index": 2, "relevance_score": 0.97,
                 "document": {"text": "Paris is the capital of France."}},
                {"index": 0, "relevance_score": 0.12,
                 "document": {"text": "Berlin is the capital of Germany."}},
            ],
            "usage": {"prompt_tokens": 0, "total_tokens": 0},
        })

    def test_backend_receives_model_query_texts_top_n(self):
        client, backend = _client([(0, 1.0)])
        self._post(client, top_n=2)
        self.assertEqual(backend.calls,
                         [(MODEL, "capital of France?", _DOCS, 2)])

    def test_dict_documents_use_text_key(self):
        client, backend = _client([(0, 1.0)])
        resp = self._post(client, documents=[{"text": "from dict"}, "plain"])
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(backend.calls[0][2], ["from dict", "plain"])
        self.assertEqual(resp.json()["results"][0]["document"],
                         {"text": "from dict"})

    def test_dict_document_without_text_is_400(self):
        client, backend = _client([(0, 1.0)])
        resp = self._post(client, documents=[{"content": "wrong key"}])
        self.assertEqual(resp.status_code, 400)
        self.assertEqual(resp.json()["error"]["code"], "unsupported_input")
        self.assertEqual(backend.calls, [])

    def test_top_n_zero_or_negative_is_422(self):
        client, _ = _client([(0, 1.0)])
        self.assertEqual(self._post(client, top_n=0).status_code, 422)
        self.assertEqual(self._post(client, top_n=-1).status_code, 422)

    def test_empty_documents_give_empty_results(self):
        client, _ = _client([])
        resp = self._post(client, documents=[])
        self.assertEqual(resp.status_code, 200)
        self.assertEqual(resp.json()["results"], [])

    def test_model_not_found_maps_to_404(self):
        client, _ = _client(ModelNotFound("org/ghost", Path("/nowhere")))
        resp = self._post(client, model="org/ghost")
        self.assertEqual(resp.status_code, 404)
        self.assertEqual(resp.json()["error"]["code"], "model_not_found")


if __name__ == "__main__":
    unittest.main()
