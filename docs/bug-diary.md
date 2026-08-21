# Engineering Bug Diary & Post-Mortem Log

This document records authentic defects, architectural regressions, and reliability anomalies encountered and resolved during the development of the Aster & Row Customer Support Agent.

---

## Bug 1 — Naive Semantic Retrieval Ranked Superseded Policy Over Active Policy

### Reproduction
Querying the raw semantic vector index with standard return window inquiries (such as `"What is the standard return window?"` or `"How long does a regular customer have to return an unused backpack?"`) returned `02-returns-policy-legacy.md` (the superseded 45-day return policy) at **Rank #1** (similarity score: `0.5082`), while the authoritative active policy `01-returns-policy-current.md` (30-day return policy) was outranked at **Rank #4** (similarity score: `0.3847`).

### Root Cause
Dense semantic search embeddings measure textual similarity rather than temporal validity or policy authority. Because the legacy policy document (`02-returns-policy-legacy.md`) contained concise return-window terminology matching user queries, standard cosine similarity favored it over the newer active policy document (`01-returns-policy-current.md`), which includes broader explanatory and operational sections. Pure vector search lacks awareness of document metadata fields like `status: superseded` vs. `status: active` and `policy_authority: official_current` vs. `policy_authority: superseded`.

### Fix
Implemented a deterministic two-stage retrieval policy in `app/rag/policy.py` (`RetrievalPolicy.apply`):
1. **Metadata-Aware Ingestion & Filtering**: Filtered candidate chunks to strictly enforce `policy_authority == "official_current"`, `status == "active"`, and `customer_answering == True`, strictly excluding `status == "superseded"` and `audience == "internal"`.
2. **Policy Precedence Weighting**: For general return queries, applied a deterministic precedence boost (+0.05) to core return authority documents (`01-returns-policy-current.md`) over adjacent product-care and warranty documents.

### Regression Test
`tests/test_retrieval_policy.py`:
- `test_standard_return_precedence_regression`
- `test_superseded_policy_excluded`

### Result
`01-returns-policy-current.md` reliably ranks #1 for all standard return queries, and legacy documents (`02-returns-policy-legacy.md`) are completely excluded from customer-facing context.

---

## Bug 2 — Source Concentration Caused Breeze Tumbler Retrieval to Miss Conflicting Source

### Reproduction
When querying `"Can I put the Breeze Tumbler in the dishwasher?"` with a retrieval limit of `top_k=2`, the dense retrieval pipeline returned two chunks exclusively from `12-breeze-tumbler-product-card.md` (scores: `0.8115` and `0.5941`), completely crowding out the contradictory guidance in `11-product-care.md` (which was displaced to rank #3 or lower).

### Root Cause
Multiple adjacent chunks from the same markdown document (`12-breeze-tumbler-product-card.md`) shared high embedding similarity with the query. Without source diversification, a single document monopolized the top-k result budget, preventing the system from observing that `11-product-care.md` explicitly specified *"main body is hand-wash only"*, whereas `12-breeze-tumbler-product-card.md` stated *"dishwasher safe on top rack"*.

### Fix
Implemented deterministic source diversification in `app/rag/policy.py` (`_diversify_sources`):
- Capped chunk representation from any single document to a maximum of 1 chunk in top-$k$ until at least one chunk from other eligible candidate documents was included.
- Preserved score-sorted ordering across diverse sources so that both `12-breeze-tumbler-product-card.md` and `11-product-care.md` are present in top-2 evidence.

### Regression Test
`tests/test_retrieval_policy.py`:
- `test_breeze_tumbler_conflict_top_2_diversification`
- `test_breeze_tumbler_conflict_sources_preserved`

### Result
Both conflicting active sources participate in the evidence context. The deterministic `EvidenceAnalyzer` correctly detects the contradiction, triggers `handoff_required = True`, and isolates citations to strictly the two conflicting documents.

---

## Bug 3 — Baseline Evaluation Mock Inadvertently Repaired Retrieval Mistakes

### Reproduction
Running `python evaluation/run_evaluation.py --baseline` initially reported a **95.0% pass rate (19/20 cases passed)**, despite the naive vector store lacking metadata filtering, precedence rules, and source diversification.

### Root Cause
The mock evaluator LLM client (`MockGeminiEvaluatorClient`) contained keyword heuristics that searched the retrieved evidence list for the *intended* authoritative document (e.g. searching for `01-returns-policy-current.md` or `09-trailplus-membership.md`) even when naive semantic search placed a legacy (Doc 02) or warranty (Doc 07) chunk at rank #1. The mock LLM layer was effectively repairing naive retrieval errors and generating correct responses from sub-optimal retrieved lists.

### Fix
Created a dedicated `BaselineMockEvaluatorClient` in `evaluation/run_evaluation.py`:
- Grounded generation and citation selection strictly and exclusively in whatever chunk naive dense search placed at **rank #1**.
- If naive search returned `02-returns-policy-legacy.md` at #1, the model generated the legacy 45-day policy.
- If naive search returned `14-internal-content-migration-notes.md` at #1, the model generated the 60-day migration note.
- Added per-case baseline retrieval audit logging displaying the top-5 raw semantic chunks and scores.

### Regression Test
`tests/test_evaluation_runner.py`:
- `test_baseline_retrieval_exposes_genuine_failures`

### Result
The corrected baseline accurately reports **14/20 passed (70.0%)**, with all 6 knowledge retrieval failures demonstrating the authentic risks of naive dense search (prompt injection vulnerability, legacy policy leakage, and missed source conflicts).

---

## Bug 4 — Live Gemini Benchmark Crashed with HTTP 429 Resource Exhaustion

### Reproduction
Executing `python evaluation/run_evaluation.py --live` against the remote `gemini-3.6-flash` API exhausted the free-tier rate limit (`generate_content_free_tier_requests`, limit 20 requests). The unhandled Google GenAI SDK `ClientError (429 RESOURCE_EXHAUSTED)` propagated to the top level, crashing the benchmark runner with a Python traceback and losing all telemetry for preceding cases.

### Root Cause
The live evaluation loop lacked boundary exception handling for remote API rate limits and quota exhaustion. When the quota was exceeded, the runner aborted immediately instead of recording completed cases and gracefully marking remaining cases as unexecuted.

### Fix
1. Implemented boundary exception handling in `evaluation/run_evaluation.py` with `LiveQuotaExhaustedError`.
2. When quota exhaustion occurs, the runner cleanly halts further API calls, preserves all previously executed case results, and marks remaining cases with `status: "not_run"` and `reason: "live_quota_exhausted"`.
3. Added exponential backoff and retry in `app/llm/client.py` (`_call_generate_with_retry`) for transient rate spikes.
4. Updated report schemas and terminal summaries to report `executed`, `passed`, `failed`, `not_run`, and `executed_pass_rate` without generating misleading aggregate percentages.

### Regression Test
`tests/test_evaluation_runner.py`:
- `test_simulated_live_quota_exhaustion_handled_gracefully`

### Result
Live evaluations that encounter rate limits terminate cleanly without tracebacks, output explicit incomplete status summaries, and persist complete telemetry to `evaluation/results/latest_live.json`.
