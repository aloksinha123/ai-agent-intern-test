"""Unit and integration tests for deterministic conflict and uncertainty detection."""

from pathlib import Path
import unittest

from app.rag.conflict import (
    ConflictingClaim,
    EvidenceAnalysis,
    EvidenceAnalyzer,
    analyze_evidence,
)
from app.rag.models import SearchResult
from app.rag.vector_store import VectorStore

REPO_ROOT = Path(__file__).resolve().parent.parent


class TestConflictDetectionUnit(unittest.TestCase):
    """Unit tests for contradiction rules and evidence sufficiency."""

    def test_generic_synthetic_conflict_detection(self):
        """Verify contradiction detection on synthetic non-corpus documents."""
        cand_a = SearchResult(
            score=0.92,
            chunk_id="device_specs.md#care",
            filename="device_specs.md",
            document_title="Device Specs",
            heading="Care",
            content="The internal sensor and all components are dishwasher safe.",
            metadata={"status": "active", "policy_authority": "official", "audience": "customer"},
        )
        cand_b = SearchResult(
            score=0.88,
            chunk_id="user_manual.md#cleaning",
            filename="user_manual.md",
            document_title="User Manual",
            heading="Cleaning",
            content="The main unit must be hand-washed with mild soap.",
            metadata={"status": "active", "policy_authority": "official", "audience": "customer"},
        )

        analysis = analyze_evidence([cand_a, cand_b], query="Can I wash this in a dishwasher?")

        self.assertEqual(analysis.status, "conflict")
        self.assertTrue(analysis.handoff_required)
        self.assertEqual(len(analysis.conflicting_claims), 1)

        claim = analysis.conflicting_claims[0]
        self.assertEqual(claim.source_a, "device_specs.md")
        self.assertEqual(claim.source_b, "user_manual.md")
        self.assertIn("dishwasher safe", claim.claim_a)
        self.assertIn("hand-washed", claim.claim_b)

    def test_consistent_evidence_no_conflict(self):
        """Verify compatible sources discussing shipping are marked consistent."""
        cand_a = SearchResult(
            score=0.90,
            chunk_id="shipping_us.md#rates",
            filename="shipping_us.md",
            document_title="US Shipping",
            heading="Rates",
            content="Standard delivery takes 3 to 5 business days across the contiguous US.",
            metadata={"status": "active", "policy_authority": "official", "audience": "customer"},
        )
        cand_b = SearchResult(
            score=0.85,
            chunk_id="membership.md#shipping",
            filename="membership.md",
            document_title="Membership",
            heading="Shipping",
            content="TrailPlus members receive free standard domestic shipping without order minimums.",
            metadata={"status": "active", "policy_authority": "official", "audience": "customer"},
        )

        analysis = analyze_evidence([cand_a, cand_b], query="How long does standard shipping take?")

        self.assertEqual(analysis.status, "consistent")
        self.assertFalse(analysis.handoff_required)
        self.assertEqual(len(analysis.conflicting_claims), 0)

    def test_insufficient_evidence_unmentioned_query_attributes(self):
        """Verify queries asking for ungrounded concepts (like vegan adhesives) trigger insufficient."""
        cand_a = SearchResult(
            score=0.45,
            chunk_id="01-returns-policy-current.md#condition",
            filename="01-returns-policy-current.md",
            document_title="Returns Policy",
            heading="Item condition",
            content="A returned item must be unused, unwashed, and in resalable condition.",
            metadata={"status": "active", "policy_authority": "official", "audience": "customer"},
        )
        cand_b = SearchResult(
            score=0.40,
            chunk_id="11-product-care.md#bags",
            filename="11-product-care.md",
            document_title="Product Care Guide",
            heading="Bags and backpacks",
            content="Spot-clean fabric bags with mild soap and cool water. Do not machine wash.",
            metadata={"status": "active", "policy_authority": "official", "audience": "customer"},
        )

        query = "Are all fabrics and adhesives in your bags vegan?"
        analysis = analyze_evidence([cand_a, cand_b], query=query)

        self.assertEqual(analysis.status, "insufficient")
        self.assertTrue(analysis.handoff_required)
        self.assertIn("vegan", analysis.reason)

    def test_insufficient_evidence_low_confidence_scores(self):
        """Verify retrieval results with low relevance scores trigger insufficient."""
        cand_low = SearchResult(
            score=0.12,
            chunk_id="misc.md#random",
            filename="misc.md",
            document_title="Misc",
            heading="Random",
            content="Unrelated text content.",
            metadata={"status": "active", "policy_authority": "official", "audience": "customer"},
        )
        analysis = analyze_evidence([cand_low], query="What is the quantum speed of the backpack?")

        self.assertEqual(analysis.status, "insufficient")
        self.assertTrue(analysis.handoff_required)


class TestConflictDetectionIntegration(unittest.TestCase):
    """End-to-end integration tests connecting VectorStore retrieval to EvidenceAnalyzer."""

    @classmethod
    def setUpClass(cls):
        storage_dir = REPO_ROOT / "storage"
        if (storage_dir / "kb_index.faiss").is_file():
            cls.store = VectorStore.load(storage_dir)
        else:
            cls.store = None

    def setUp(self):
        if self.store is None:
            self.skipTest("Persisted FAISS index not found; run scripts/build_index.py first.")

    def test_breeze_tumbler_dishwasher_conflict_end_to_end(self):
        """Query: 'Can I put the Breeze Tumbler in the dishwasher?' must trigger conflict status and handoff."""
        query = "Can I put the Breeze Tumbler in the dishwasher?"
        results = self.store.retrieve(query, top_k=5, mode="customer")

        analysis = analyze_evidence(results, query=query)

        self.assertEqual(analysis.status, "conflict")
        self.assertTrue(analysis.handoff_required)
        self.assertGreater(len(analysis.conflicting_claims), 0)

        # Check source provenance
        claim = analysis.conflicting_claims[0]
        filenames = {claim.source_a, claim.source_b}
        self.assertIn("12-breeze-tumbler-product-card.md", filenames)
        self.assertIn("11-product-care.md", filenames)

        # Confirm JSON serialization
        log_payload = analysis.to_dict()
        self.assertEqual(log_payload["status"], "conflict")
        self.assertTrue(log_payload["handoff_required"])

    def test_standard_return_window_consistent(self):
        """Query: 'What is the standard return window?' must be consistent without conflict."""
        query = "What is the standard return window?"
        results = self.store.retrieve(query, top_k=5, mode="customer")

        analysis = analyze_evidence(results, query=query)

        self.assertEqual(analysis.status, "consistent")
        self.assertFalse(analysis.handoff_required)
        self.assertEqual(len(analysis.conflicting_claims), 0)
        self.assertEqual(results[0].filename, "01-returns-policy-current.md")

    def test_vegan_materials_insufficient_end_to_end(self):
        """Query: 'Are all fabrics and adhesives in your bags vegan?' must trigger insufficient status."""
        query = "Are all fabrics and adhesives in your bags vegan?"
        results = self.store.retrieve(query, top_k=5, mode="customer")

        analysis = analyze_evidence(results, query=query)

        self.assertEqual(analysis.status, "insufficient")
        self.assertTrue(analysis.handoff_required)
        self.assertIn("vegan", analysis.reason)


if __name__ == "__main__":
    unittest.main()
