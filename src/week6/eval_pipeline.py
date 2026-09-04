import json
import os
from week6_pipeline import ask_rag,retrieve, load_golden_set, CORPUS, index

def eval_hit_rate(golden, index, k=3):
    """
    Compute hit rate: fraction of questions where the correct source_doc
    appears in the retrieved top-K chunks.
    """
    total = len(golden)
    hits = 0
    detailed = []

    for item in golden:
        q = item["question"]
        expected_doc = item["source_doc"]

        #retrieved = retrieve(q, index, k=k)
        # Run RAG
        rag_out=ask_rag(q, index,k=k)
        retrieved_docs = {chunk["source_id"] for chunk in rag_out["retrieved"]}

        hit = expected_doc in retrieved_docs
        if hit:
            hits += 1

        detailed.append({
            "id": item["id"],
            "question": q,
            "expected_doc": expected_doc,
            "retrieved_docs": list(retrieved_docs),
            "hit": hit,
            "rag_answer": rag_out["answer"],     # ← actual RAG answer
            "sources": rag_out["sources"],       # ← chunk_ids used
            "tokens_in": rag_out["tokens_in"],
            "tokens_out": rag_out["tokens_out"]
        })

    return hits / total, detailed


def main():
    golden_path = "golden/nhanes_golden_set.jsonl"
    golden = load_golden_set(golden_path)

    hit_rate, detailed = eval_hit_rate(golden, index, k=3)

    print(f"Total questions: {len(golden)}")
    print(f"Hit rate@3: {hit_rate:.3f}")

    # Optional: write detailed results
    with open("eval_results_pipeline.jsonl", "w", encoding="utf-8") as f:
        for row in detailed:
            f.write(json.dumps(row) + "\n")

    print("Saved eval_results.jsonl")


if __name__ == "__main__":
    main()
