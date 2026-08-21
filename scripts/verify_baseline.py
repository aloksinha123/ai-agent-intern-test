import json
from pathlib import Path
import sys

REPO_ROOT = Path(__file__).resolve().parent.parent
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from app.rag.vector_store import VectorStore
from app.rag.embeddings import embed_query

BASELINE_FILE = Path("evaluation/results/latest_baseline.json")
VISIBLE_CASES = Path("evaluation/visible-cases.json")
CUSTOM_CASES = Path("evaluation/custom-cases.json")


def main() -> None:
    if not BASELINE_FILE.is_file():
        print(f"[ERROR] Baseline artifact not found at {BASELINE_FILE}")
        return

    with open(BASELINE_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    print("=" * 80)
    print("STORED BASELINE ARTIFACT TELEMETRY VERIFICATION")
    print("=" * 80)
    print(f"Mode               : {data.get('mode')}")
    print(f"Timestamp          : {data.get('timestamp')}")
    print(f"Total Cases        : {data.get('total_cases')}")
    print(f"Total Passed       : {data.get('total_passed')}")
    print(f"Overall Percentage : {data.get('overall_percentage')}%\n")

    cases_dict = {c["case_id"]: c for c in data.get("cases", [])}
    failed_cases = [c for c in data.get("cases", []) if not c.get("passed")]

    all_cases_meta = []
    for cf in [VISIBLE_CASES, CUSTOM_CASES]:
        with open(cf, "r", encoding="utf-8") as f:
            all_cases_meta.extend(json.load(f).get("cases", []))
    meta_by_id = {c["id"]: c for c in all_cases_meta}

    print(f"FAILED CASES AUDIT ({len(failed_cases)} total failures):")
    print("-" * 80)

    store = VectorStore.load()

    for idx, c in enumerate(failed_cases, 1):
        cid = c["case_id"]
        cat = c["category"]
        print(f"\n[{idx}/{len(failed_cases)}] Case: {cid} (Category: {cat})")
        print("  Failures Recorded:")
        for fail in c["failures"]:
            print(f"    |-- {fail}")
        print(f"  Final Answer Text: {c.get('final_text')}")
        print(f"  Citations: {c.get('citations')}")

        meta = meta_by_id.get(cid, {})
        messages = meta.get("messages", [])
        if messages:
            last_msg = messages[-1].get("content", "")
            print(f"  Raw Query: \"{last_msg}\"")
            vec = embed_query(last_msg)
            raw_top5 = store.search(vec, top_k=5)
            print("  Exact Raw Top-5 Retrieval at Execution Time:")
            for r_idx, r in enumerate(raw_top5, 1):
                print(f"    {r_idx}. {r.filename:<38} | {r.heading:<28} | score: {r.score:.4f}")

    print("\n" + "=" * 80)
    print("SPECIFIC VERIFICATION: standard-return-window")
    print("=" * 80)
    std_meta = meta_by_id.get("standard-return-window", {})
    std_query = std_meta.get("messages", [{}])[0].get("content", "")
    print(f"Query: \"{std_query}\"")
    std_vec = embed_query(std_query)
    std_results = store.search(std_vec, top_k=5)
    for idx, r in enumerate(std_results, 1):
        is_target = " <<< [TARGET EXPECTED]" if r.filename == "01-returns-policy-current.md" else ""
        print(f"  Rank {idx}: {r.filename:<38} | {r.heading:<25} | score: {r.score:.4f}{is_target}")

    rank_doc01 = next((i for i, r in enumerate(std_results, 1) if r.filename == "01-returns-policy-current.md"), None)
    print(f"\n  -> Exact Rank 1: {std_results[0].filename} ({std_results[0].heading}, score {std_results[0].score:.4f})")
    print(f"  -> Exact Rank 2: {std_results[1].filename} ({std_results[1].heading}, score {std_results[1].score:.4f})")
    print(f"  -> Exact Rank of 01-returns-policy-current.md: Rank {rank_doc01} ({std_results[rank_doc01-1].heading}, score {std_results[rank_doc01-1].score:.4f})")
    print("=" * 80)


if __name__ == "__main__":
    main()
