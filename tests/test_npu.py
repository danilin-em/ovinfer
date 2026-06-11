"""Tests for ovinfer.npu: the conservative NPU-incompatibility preflight.

reject_reason(model_dir, kind) returns a human-readable reason string when a
(model, kind) is known to fail on NPU, or None when it should be left alone.
The heuristic is deliberately conservative: it only flags cases verified
empirically (group-wise int4 LLMs fail to compile; embeddings and rerankers
fail at inference due to NPU's static-shape requirement).
"""

import tempfile
import unittest
from pathlib import Path

from ovinfer import npu


def _model_dir(rt_info_compression: str | None) -> Path:
    """Create a temp model dir with a minimal openvino_model.xml.

    rt_info_compression is the inner XML of <weight_compression>, or None to
    write an IR with no nncf weight-compression rt_info at all (e.g. fp16).
    """
    d = Path(tempfile.mkdtemp())
    block = (
        f"<rt_info><nncf><weight_compression>{rt_info_compression}"
        "</weight_compression></nncf></rt_info>"
        if rt_info_compression is not None
        else ""
    )
    (d / "openvino_model.xml").write_text(
        f'<?xml version="1.0"?>\n<net name="m" version="11">{block}</net>'
    )
    return d


# Real nncf rt_info shape: <mode> sits next to a <backup_mode> the parser must
# NOT pick up, plus <group_size>.
_GROUPWISE = (
    '<mode value="int4_asym" /><backup_mode value="int8_asym" />'
    '<group_size value="128" /><ratio value="0.8" />'
)
_CHANNELWISE = (
    '<mode value="int4_sym" /><backup_mode value="int8_asym" />'
    '<group_size value="-1" /><ratio value="1.0" />'
)
_INT8 = '<mode value="int8_asym" /><group_size value="-1" />'


class RejectReasonTest(unittest.TestCase):
    def test_embedding_on_npu_is_flagged(self):
        # Compiles on NPU but fails at inference (NPUW static shape); the only
        # case the compiler itself does not catch, so the preflight must.
        reason = npu.reject_reason(_model_dir(_CHANNELWISE), "embedding")
        self.assertIsNotNone(reason)
        self.assertIn("static", reason.lower())

    def test_rerank_on_npu_is_flagged(self):
        # Cross-encoder rerankers share the embedding failure mode: dynamic
        # shapes vs NPU's static-shape requirement, failing at request time.
        reason = npu.reject_reason(_model_dir(_INT8), "rerank")
        self.assertIsNotNone(reason)
        self.assertIn("static", reason.lower())

    def test_groupwise_int4_llm_is_flagged_with_group_size(self):
        reason = npu.reject_reason(_model_dir(_GROUPWISE), "llm")
        self.assertIsNotNone(reason)
        self.assertIn("128", reason)            # reports the actual group size
        self.assertIn("channel-wise", reason)    # names the fix

    def test_channelwise_int4_llm_is_not_flagged(self):
        self.assertIsNone(npu.reject_reason(_model_dir(_CHANNELWISE), "llm"))

    def test_int8_llm_is_not_flagged(self):
        # Not the int4 group-wise case; we have no evidence it fails — stay quiet.
        self.assertIsNone(npu.reject_reason(_model_dir(_INT8), "llm"))

    def test_uncompressed_llm_is_not_flagged(self):
        self.assertIsNone(npu.reject_reason(_model_dir(None), "llm"))

    def test_vlm_on_npu_is_not_flagged(self):
        # VLM on NPU is unverified on this stack (og puts only the language
        # model on NPU); the conservative preflight stays silent.
        self.assertIsNone(npu.reject_reason(_model_dir(_CHANNELWISE), "vlm"))


class WarnIfIncompatibleTest(unittest.TestCase):
    """warn_if_incompatible logs a single WARNING on the 'ovinfer' logger for an
    NPU placement that reject_reason flags, and stays silent otherwise."""

    def test_warns_for_groupwise_llm_on_npu(self):
        with self.assertLogs("ovinfer", level="WARNING") as cm:
            npu.warn_if_incompatible("org/m", _model_dir(_GROUPWISE), "llm", "NPU")
        self.assertEqual(len(cm.records), 1)
        msg = cm.records[0].getMessage()
        self.assertIn("org/m", msg)
        self.assertIn("group_size=128", msg)

    def test_warns_for_embedding_on_npu(self):
        with self.assertLogs("ovinfer", level="WARNING") as cm:
            npu.warn_if_incompatible("org/e", _model_dir(_INT8), "embedding", "NPU")
        self.assertEqual(len(cm.records), 1)
        self.assertIn("org/e", cm.records[0].getMessage())

    def test_silent_for_channelwise_llm_on_npu(self):
        with self.assertNoLogs("ovinfer", level="WARNING"):
            npu.warn_if_incompatible("org/m", _model_dir(_CHANNELWISE), "llm", "NPU")

    def test_silent_for_groupwise_llm_off_npu(self):
        # Same incompatible model, but not on NPU — nothing to warn about.
        with self.assertNoLogs("ovinfer", level="WARNING"):
            npu.warn_if_incompatible("org/m", _model_dir(_GROUPWISE), "llm", "GPU")


if __name__ == "__main__":
    unittest.main()
