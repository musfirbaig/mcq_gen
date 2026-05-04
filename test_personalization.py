import os
import sys
import unittest
from types import SimpleNamespace
from unittest.mock import patch


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
REPO_ROOT = os.path.dirname(ROOT_DIR)
if REPO_ROOT not in sys.path:
    sys.path.insert(0, REPO_ROOT)
if ROOT_DIR not in sys.path:
    sys.path.insert(0, ROOT_DIR)

import server  # noqa: E402
import shared_learner_memory as learner_memory  # noqa: E402


class FakeMatch:
    def __init__(self, score, metadata):
        self.score = score
        self.metadata = metadata


class FakeIndex:
    def query(self, vector, top_k, filter, include_metadata):
        _ = (vector, top_k, include_metadata)
        if "sub_topic_key" in filter:
            return SimpleNamespace(matches=[])
        if "topic_key" in filter:
            return SimpleNamespace(
                matches=[
                    FakeMatch(
                        0.67,
                        {
                            "session_id": "sess_topic_1",
                            "excerpt": "Weakness in integration by parts setup.",
                            "weak_summary": "Learner confuses selecting u/dv and loses constants while integrating by parts.",
                            "focus_areas": '["Select u/dv strategically", "Track constants and signs carefully"]',
                            "next_mcq_guidance": '["Generate by-parts setup questions first"]',
                            "weak_concepts": '["Forgets to pick u and dv correctly"]',
                            "last_score": 48,
                        },
                    )
                ]
            )
        return SimpleNamespace(
            matches=[
                FakeMatch(
                    0.52,
                    {
                        "session_id": "sess_global_1",
                        "excerpt": "Generic algebra mistakes under timed pressure.",
                        "weak_concepts": '["Sign errors in simplification"]',
                        "last_score": 55,
                    },
                )
            ]
        )


class PersonalizationTests(unittest.TestCase):
    def test_identity_resolution_prefers_header(self):
        resolved, source = server._resolve_user_id("form-user", "header-user")
        self.assertEqual(resolved, "header-user")
        self.assertEqual(source, "header")

    def test_identity_resolution_falls_back_to_form(self):
        resolved, source = server._resolve_user_id("form-user", None)
        self.assertEqual(resolved, "form-user")
        self.assertEqual(source, "form")

    def test_query_memory_bundle_uses_tiered_fallback(self):
        with patch.object(learner_memory, "is_configured", return_value=True), patch.object(
            learner_memory, "_client", return_value=(object(), FakeIndex())
        ), patch.object(learner_memory, "embed_query", return_value=[0.1, 0.2, 0.3]), patch.dict(
            os.environ,
            {"RAG_MIN_HITS_FOR_CONFIDENCE": "1"},
            clear=False,
        ):
            bundle = learner_memory.query_memory_bundle(
                user_id="user-123",
                topic_key="calculus",
                sub_topic_key="integration",
                query_text="integration by parts",
                top_k=8,
                min_score=0.1,
            )

        self.assertGreaterEqual(bundle["memory_hits"], 1)
        self.assertGreaterEqual(bundle["summary_hits"], 1)
        self.assertEqual(bundle["retrieval_tier"], "topic_lane")
        self.assertIn("user:user-123", bundle["lane_used"])
        self.assertTrue(bundle["learner_context"])
        self.assertIn("summary", bundle["learner_context"])

    def test_feedback_summary_helper_returns_structured_payload(self):
        class FakeResponse:
            content = (
                '{"summary":"Needs better chain-rule mapping.",'
                '"focus_areas":["Map inner and outer functions"],'
                '"common_mistakes":["Drops derivative of inner function"],'
                '"next_mcq_guidance":["Use mixed chain-rule templates"]}'
            )

        class FakeLLM:
            def invoke(self, _messages):
                return FakeResponse()

        with patch.object(server, "ChatOpenAI", return_value=FakeLLM()):
            result = server._summarize_weak_concepts_for_memory(
                subject="Calculus",
                chapter="Integration",
                generation_query="chain rule weakness",
                weak_concepts=["Missed inner derivative"],
                correct=1,
                wrong=3,
                graded=4,
                last_score=25.0,
                model="dummy-model",
            )

        self.assertTrue(result["summary"])
        self.assertGreaterEqual(len(result["focus_areas"]), 1)
        self.assertGreaterEqual(len(result["next_mcq_guidance"]), 1)


if __name__ == "__main__":
    unittest.main()
