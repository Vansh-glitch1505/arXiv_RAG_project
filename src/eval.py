import os
import json
from dotenv import load_dotenv
from answer import answer_question

load_dotenv()

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EVAL_SET_PATH = os.path.join(BASE_DIR, "evals", "eval_set.jsonl")
RESULTS_PATH = os.path.join(BASE_DIR, "evals", "eval_results.json")


def load_eval_set() -> list[dict]:
    with open(EVAL_SET_PATH, "r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def check_retrieval(expected_paper_id: str, docs: list) -> bool:
    """Did the reranked chunks include at least one chunk from the correct paper?"""
    retrieved_paper_ids = {doc.metadata.get("paper_id") for doc in docs}
    return expected_paper_id in retrieved_paper_ids


def check_facts(answer_text: str, expected_facts: list[str]) -> tuple[int, int]:
    """How many of the expected facts appear (case-insensitive substring) in the answer?"""
    answer_lower = answer_text.lower()
    hits = sum(1 for fact in expected_facts if fact.lower() in answer_lower)
    return hits, len(expected_facts)


def run_eval():
    eval_set = load_eval_set()
    results = []

    retrieval_correct = 0
    total_fact_hits = 0
    total_facts = 0

    for item in eval_set:
        print(f"\n[{item['id']}] {item['question']}")
        answer_text, docs = answer_question(item["question"])

        retrieval_ok = check_retrieval(item["expected_paper_id"], docs)
        fact_hits, fact_total = check_facts(answer_text, item["expected_facts"])

        retrieval_correct += int(retrieval_ok)
        total_fact_hits += fact_hits
        total_facts += fact_total

        print(f"  Retrieval correct paper: {'YES' if retrieval_ok else 'NO'}")
        print(f"  Facts found: {fact_hits}/{fact_total}")

        results.append({
            "id": item["id"],
            "question": item["question"],
            "expected_paper_id": item["expected_paper_id"],
            "retrieval_correct": retrieval_ok,
            "facts_found": fact_hits,
            "facts_total": fact_total,
            "answer": answer_text
        })

    n = len(eval_set)
    print("\n" + "=" * 50)
    print(f"RETRIEVAL ACCURACY: {retrieval_correct}/{n} ({100 * retrieval_correct / n:.1f}%)")
    print(f"FACT RECALL: {total_fact_hits}/{total_facts} ({100 * total_fact_hits / total_facts:.1f}%)")
    print("=" * 50)

    with open(RESULTS_PATH, "w", encoding="utf-8") as f:
        json.dump({
            "retrieval_accuracy": f"{retrieval_correct}/{n}",
            "fact_recall": f"{total_fact_hits}/{total_facts}",
            "details": results
        }, f, indent=2)
    print(f"\nFull results saved to {RESULTS_PATH}")


if __name__ == "__main__":
    run_eval()