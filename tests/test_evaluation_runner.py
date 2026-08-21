"""Unit tests for the evaluation runner and assertion mechanics."""

import hashlib
import json
from pathlib import Path
import unittest
from unittest.mock import MagicMock

from evaluation.run_evaluation import (
    CUSTOM_CASES_PATH,
    RESULTS_DIR,
    VISIBLE_CASES_PATH,
    evaluate_case,
    normalize_text,
    run_benchmark,
)
from app.agent.orchestrator import AgentOrchestrator, AgentTurnResponse


class TestEvaluationRunner(unittest.TestCase):
    """Test evaluation runner loading, assertion logic, and report generation."""

    @classmethod
    def setUpClass(cls):
        with open(VISIBLE_CASES_PATH, "rb") as f:
            cls.initial_visible_hash = hashlib.sha256(f.read()).hexdigest()

    def test_visible_cases_loaded_without_mutation(self):
        """Visible cases file exists, parses correctly, and remains unmodified."""
        self.assertTrue(VISIBLE_CASES_PATH.is_file())
        with open(VISIBLE_CASES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        cases = data.get("cases", [])
        self.assertEqual(len(cases), 15)

        with open(VISIBLE_CASES_PATH, "rb") as f:
            current_hash = hashlib.sha256(f.read()).hexdigest()
        self.assertEqual(self.initial_visible_hash, current_hash)

    def test_custom_cases_loaded(self):
        """Custom cases file exists and contains exactly 5 cases."""
        self.assertTrue(CUSTOM_CASES_PATH.is_file())
        with open(CUSTOM_CASES_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)

        cases = data.get("cases", [])
        self.assertEqual(len(cases), 5)

    def test_text_normalization(self):
        """Whitespace and casing normalization works robustly."""
        self.assertEqual(normalize_text("  30  Calendar   DAYS  "), "30 calendar days")

    def test_assertion_pass_behavior(self):
        """Mock turn response that satisfies all expectations passes evaluation."""
        mock_orchestrator = MagicMock(spec=AgentOrchestrator)
        mock_orchestrator.process_turn.return_value = AgentTurnResponse(
            session_id="test_sess",
            intent="knowledge",
            text="The standard return window is 30 calendar days from delivery.",
            citations=["01-returns-policy-current.md — Standard return window"],
            handoff_required=False,
            tool_called=False,
        )

        sample_case = {
            "id": "sample-pass",
            "category": "retrieval",
            "messages": [{"role": "user", "content": "Return window query"}],
            "expect": {
                "must_include": ["30 calendar days"],
                "must_not_include": ["60 days"],
                "required_sources": ["01-returns-policy-current.md"],
                "tool": "not_called",
                "handoff": False,
            },
        }

        passed, failures, telemetry = evaluate_case(sample_case, mock_orchestrator)
        self.assertTrue(passed)
        self.assertEqual(len(failures), 0)
        self.assertEqual(telemetry["case_id"], "sample-pass")

    def test_assertion_fail_behavior(self):
        """Mock turn response that violates assertions correctly records failure reasons."""
        mock_orchestrator = MagicMock(spec=AgentOrchestrator)
        mock_orchestrator.process_turn.return_value = AgentTurnResponse(
            session_id="test_sess",
            intent="knowledge",
            text="Here is our 60 days return policy.",
            citations=["02-returns-policy-legacy.md — Old Policy"],
            handoff_required=False,
            tool_called=False,
        )

        sample_case = {
            "id": "sample-fail",
            "category": "retrieval",
            "messages": [{"role": "user", "content": "Return query"}],
            "expect": {
                "must_include": ["30 calendar days"],
                "must_not_include": ["60 days"],
                "required_sources": ["01-returns-policy-current.md"],
                "forbidden_sources_as_authority": ["02-returns-policy-legacy.md"],
                "tool": "not_called",
                "handoff": False,
            },
        }

        passed, failures, telemetry = evaluate_case(sample_case, mock_orchestrator)
        self.assertFalse(passed)
        self.assertGreaterEqual(len(failures), 3)

    def test_benchmark_mock_run_produces_valid_report(self):
        """Running benchmark in mock mode produces latest_mock.json with stats."""
        report = run_benchmark(cases_files=[VISIBLE_CASES_PATH, CUSTOM_CASES_PATH], mode="mock")
        self.assertEqual(report["mode"], "mock")
        self.assertEqual(report["total_cases"], 20)
        self.assertEqual(report["total_passed"], 20)
        self.assertEqual(report["total_not_run"], 0)
        self.assertEqual(report["status"], "completed")
        self.assertEqual(report["executed_pass_rate"], 100.0)

        mock_out = RESULTS_DIR / "latest_mock.json"
        self.assertTrue(mock_out.is_file())

    def test_simulated_live_quota_exhaustion_handled_gracefully(self):
        """Simulated HTTP 429 quota exhaustion stops benchmark cleanly and records not_run cases."""
        from unittest.mock import patch
        from evaluation.run_evaluation import LiveQuotaExhaustedError

        # Simulate quota exhaustion on case 2
        call_count = [0]

        def mock_evaluate(case, orchestrator):
            call_count[0] += 1
            if call_count[0] == 1:
                return True, [], {"case_id": case.get("id"), "category": case.get("category"), "passed": True, "failures": [], "elapsed_ms": 10.0}
            raise LiveQuotaExhaustedError("Gemini API quota/rate-limit exhausted: 429 RESOURCE_EXHAUSTED")

        with patch("evaluation.run_evaluation.evaluate_case", side_effect=mock_evaluate):
            report = run_benchmark(cases_files=[VISIBLE_CASES_PATH], mode="mock")

            self.assertTrue(report["quota_exhausted"])
            self.assertEqual(report["status"], "incomplete_due_to_quota")
            self.assertEqual(report["total_cases"], 15)
            self.assertEqual(report["total_executed"], 1)
            self.assertEqual(report["total_passed"], 1)
            self.assertEqual(report["total_not_run"], 14)
            self.assertEqual(report["executed_pass_rate"], 100.0)

            # Check not_run cases are clearly marked
            not_run_cases = [c for c in report["cases"] if c.get("status") == "not_run"]
            self.assertEqual(len(not_run_cases), 14)
            for nrc in not_run_cases:
                self.assertEqual(nrc["reason"], "live_quota_exhausted")
                self.assertFalse(nrc["passed"])


if __name__ == "__main__":
    unittest.main()
