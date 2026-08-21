"""Unit and regression tests for metadata-aware retrieval policy and precedence."""

from pathlib import Path
import unittest

from app.rag.loader import load_knowledge_base
from app.rag.models import Chunk, SearchResult
from app.rag.policy import RetrievalPolicy
from app.rag.splitter import split_documents
from app.rag.vector_store import VectorStore

REPO_ROOT = Path(__file__).resolve().parent.parent
KB_DIR = REPO_ROOT / "knowledge-base"


class TestRetrievalPolicyUnit(unittest.TestCase):
    """Unit tests for metadata eligibility rules."""

    def test_active_official_customer_eligible(self):
        """Active official customer doc is eligible even if customer_answering is omitted."""
        meta = {
            "document_id": "RET-2026-01",
            "status": "active",
            "policy_authority": "official",
            "audience": "customer",
        }
        eligible, reason = RetrievalPolicy.is_eligible(meta, mode="customer")
        self.assertTrue(eligible)
        self.assertEqual(reason, "eligible_customer_evidence")

    def test_superseded_policy_excluded(self):
        """Superseded doc (Doc 02) is strictly excluded in customer mode."""
        meta = {
            "document_id": "RET-2024-01",
            "status": "superseded",
            "policy_authority": "official",
            "audience": "customer",
        }
        eligible, reason = RetrievalPolicy.is_eligible(meta, mode="customer")
        self.assertFalse(eligible)
        self.assertEqual(reason, "excluded_superseded_policy")

    def test_draft_scratchpad_excluded(self):
        """Draft scratchpad (Doc 14) with policy_authority=none is excluded."""
        meta = {
            "document_id": "MIG-TEST-04",
            "status": "draft",
            "policy_authority": "none",
            "audience": "internal",
            "customer_answering": False,
        }
        eligible, reason = RetrievalPolicy.is_eligible(meta, mode="customer")
        self.assertFalse(eligible)
        self.assertIn("excluded", reason)

    def test_explicit_customer_answering_false_excluded(self):
        """Doc with customer_answering=False is excluded even if other fields match."""
        meta = {
            "status": "active",
            "policy_authority": "official",
            "audience": "customer",
            "customer_answering": False,
        }
        eligible, reason = RetrievalPolicy.is_eligible(meta, mode="customer")
        self.assertFalse(eligible)
        self.assertEqual(reason, "excluded_customer_answering_false")

    def test_internal_escalation_mode_separation(self):
        """Doc 13 (internal audience) is excluded in customer mode, but eligible in internal mode."""
        meta = {
            "document_id": "SUP-2026-01",
            "status": "active",
            "policy_authority": "official",
            "audience": "internal",
        }
        # Customer mode: excluded
        cust_eligible, cust_reason = RetrievalPolicy.is_eligible(meta, mode="customer")
        self.assertFalse(cust_eligible)
        self.assertEqual(cust_reason, "excluded_non_customer_audience_internal")

        # Internal mode: eligible
        int_eligible, int_reason = RetrievalPolicy.is_eligible(meta, mode="internal")
        self.assertTrue(int_eligible)
        self.assertEqual(int_reason, "eligible_internal_evidence")


class TestRetrievalPolicyIntegration(unittest.TestCase):
    """End-to-end regression tests using persisted or mock VectorStore."""

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

    def test_standard_return_precedence_regression(self):
        """Query 1: 'What is the standard return window?' must return Doc 01 as #1."""
        results = self.store.retrieve("What is the standard return window?", top_k=5, mode="customer")

        self.assertGreater(len(results), 0)
        top_res = results[0]
        self.assertEqual(top_res.filename, "01-returns-policy-current.md")
        self.assertEqual(top_res.heading, "Standard return window")
        self.assertIn("30 calendar days", top_res.content)

        # Ensure superseded Doc 02 is completely absent from customer evidence
        retrieved_files = [r.filename for r in results]
        self.assertNotIn("02-returns-policy-legacy.md", retrieved_files)

    def test_trailplus_membership_specificity_regression(self):
        """Query 2: TrailPlus query must return Doc 09 > Return window as #1."""
        results = self.store.retrieve(
            "How long do TrailPlus members have to return an item?",
            top_k=5,
            mode="customer",
        )

        self.assertGreater(len(results), 0)
        top_res = results[0]
        self.assertEqual(top_res.filename, "09-trailplus-membership.md")
        self.assertEqual(top_res.heading, "Return window")
        self.assertIn("45-calendar-day return window", top_res.content)

    def test_breeze_tumbler_conflict_sources_preserved(self):
        """Query 4: Both conflicting active sources (Doc 11 and Doc 12) must remain eligible."""
        results = self.store.retrieve(
            "Can I put the Breeze Tumbler in the dishwasher?",
            top_k=5,
            mode="customer",
        )

        retrieved_ids = [r.chunk_id for r in results]
        has_doc_11 = any("11-product-care.md" in cid for cid in retrieved_ids)
        has_doc_12 = any("12-breeze-tumbler-product-card.md" in cid for cid in retrieved_ids)
        self.assertTrue(has_doc_11, "Expected 11-product-care.md to be retrieved.")
        self.assertTrue(has_doc_12, "Expected 12-breeze-tumbler-product-card.md to be retrieved.")

    def test_breeze_tumbler_conflict_top_2_diversification(self):
        """Query 4 with top_k=2 must include BOTH Doc 12 and Doc 11 (prevent same-doc crowding)."""
        results = self.store.retrieve(
            "Can I put the Breeze Tumbler in the dishwasher?",
            top_k=2,
            mode="customer",
        )

        self.assertEqual(len(results), 2)
        filenames = [r.filename for r in results]
        self.assertIn("12-breeze-tumbler-product-card.md", filenames)
        self.assertIn("11-product-care.md", filenames)

    def test_general_source_diversification_synthetic(self):
        """Verify general diversification across arbitrary documents without single-doc dominance."""
        # Create synthetic chunks where Doc A dominates top scores
        cand1 = SearchResult(score=0.95, chunk_id="docA#1", filename="docA.md", document_title="Doc A", heading="H1", content="c1", final_score=0.95)
        cand2 = SearchResult(score=0.90, chunk_id="docA#2", filename="docA.md", document_title="Doc A", heading="H2", content="c2", final_score=0.90)
        cand3 = SearchResult(score=0.85, chunk_id="docA#3", filename="docA.md", document_title="Doc A", heading="H3", content="c3", final_score=0.85)
        cand4 = SearchResult(score=0.80, chunk_id="docB#1", filename="docB.md", document_title="Doc B", heading="H1", content="c4", final_score=0.80)
        cand5 = SearchResult(score=0.75, chunk_id="docC#1", filename="docC.md", document_title="Doc C", heading="H1", content="c5", final_score=0.75)

        candidates = [cand1, cand2, cand3, cand4, cand5]

        # Diversifying with top_k=3 should select top from Doc A, Doc B, Doc C (not all 3 from Doc A)
        diversified = RetrievalPolicy.diversify_sources(candidates, top_k=3)
        self.assertEqual(len(diversified), 3)
        selected_files = [r.filename for r in diversified]
        self.assertEqual(selected_files, ["docA.md", "docB.md", "docC.md"])
        self.assertEqual(diversified[0].chunk_id, "docA#1")
        self.assertEqual(diversified[1].chunk_id, "docB#1")
        self.assertEqual(diversified[2].chunk_id, "docC#1")

        # Diversifying with top_k=4 should include Doc A, Doc B, Doc C, then next best from Doc A
        diversified_4 = RetrievalPolicy.diversify_sources(candidates, top_k=4)
        self.assertEqual(len(diversified_4), 4)
        selected_4_files = [r.filename for r in diversified_4]
        self.assertEqual(selected_4_files, ["docA.md", "docB.md", "docC.md", "docA.md"])
        self.assertEqual(diversified_4[3].chunk_id, "docA#2")

    def test_sixty_day_query_excludes_unapproved_notes(self):
        """Query 5: 60-day query must exclude Doc 14 draft scratchpad and Doc 02."""
        results = self.store.retrieve(
            "Do you offer a 60 day return policy?",
            top_k=5,
            mode="customer",
        )

        retrieved_files = [r.filename for r in results]
        self.assertNotIn("14-internal-content-migration-notes.md", retrieved_files)
        self.assertNotIn("02-returns-policy-legacy.md", retrieved_files)
        # Authoritative active return policies should be returned
        self.assertIn("01-returns-policy-current.md", retrieved_files)

    def test_international_shipping_destinations_remains_top(self):
        """Query 3: International shipping destinations query returns Doc 06 as #1."""
        results = self.store.retrieve(
            "Do you ship internationally?",
            top_k=5,
            mode="customer",
        )

        self.assertGreater(len(results), 0)
        self.assertEqual(results[0].filename, "06-international-shipping.md")
        self.assertEqual(results[0].heading, "Supported destinations")
        self.assertIn("Canada", results[0].content)


if __name__ == "__main__":
    unittest.main()
