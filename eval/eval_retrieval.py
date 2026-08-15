"""Retrieval eval: precision@k / recall@k for relevant_companies() against
eval/gold_queries.json.

Only meaningful once you've run pipeline/ingest.py on real, ingested
reports -- this queries the live ChromaDB collection app/retriever.py
reads from. Every gold query names its expected company/companies
explicitly (see gold_queries.json's _readme) so "expected" is an objective
label, not something this script guessed.

Run: .venv/bin/python eval/eval_retrieval.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.retriever import relevant_companies

GOLD_PATH = Path(__file__).parent / "gold_queries.json"


def precision_recall(expected: set[str], retrieved: list[str]) -> tuple[float, float]:
    retrieved_set = set(retrieved)
    if not retrieved_set:
        return (0.0, 0.0)
    hits = expected & retrieved_set
    precision = len(hits) / len(retrieved_set)
    recall = len(hits) / len(expected) if expected else 1.0
    return (precision, recall)


def main() -> None:
    gold = json.loads(GOLD_PATH.read_text())["queries"]

    print(f"Retrieval eval: {len(gold)} gold queries\n")
    precisions, recalls = [], []
    for item in gold:
        expected = set(item["expected_companies"])
        retrieved = relevant_companies(item["query"], n_results=8, max_companies=len(expected) + 2)
        p, r = precision_recall(expected, retrieved)
        precisions.append(p)
        recalls.append(r)

        status = "OK" if r == 1.0 else "MISS"
        print(f"[{status}] {item['query']!r}")
        print(f"    expected:  {sorted(expected)}")
        print(f"    retrieved: {retrieved}")
        print(f"    precision={p:.2f}  recall={r:.2f}\n")

    n = len(gold)
    print(f"Mean precision: {sum(precisions)/n:.2%}")
    print(f"Mean recall:    {sum(recalls)/n:.2%}")
    print(f"Perfect recall: {sum(1 for r in recalls if r == 1.0)}/{n} queries")


if __name__ == "__main__":
    main()
