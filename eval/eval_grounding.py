"""Generation eval: citation grounding + a citation-coverage faithfulness
proxy, for eval/gold_queries.json run through app/llm.py's ask().

What this checks and why it's still worth checking even though the Citations
API extracts citations from the attached PDF (so it can't hallucinate a
citation's *location*): the check here is on *this project's own*
offset-correction pipeline (pipeline/pdf_utils.py's page-offset stamping,
app/llm.py's exclusive-end-page fix), not on the model. A bug in either of
those would make a citation's reported page wrong even though the API's own
citation extraction is correct -- this test re-derives the local page,
re-reads the actual attached PDF, and confirms cited_text is really there.

Also reports a citation-coverage number as a faithfulness proxy: what share
of the answer's characters fall inside a cited span. This is NOT the same as
verifying every *claim* is faithful (that needs a human or a judge model,
same as any RAG system with free-text sources) -- it only tells you how much
of the answer is backed by a citation at all versus how much is unbacked
prose (synthesis, transitions, or unsupported claims).

Also reports token usage and $ cost per query. This matters specifically
for this architecture: unlike chunk-based retrieval, every query attaches
the *full* trimmed PDF for each matched company, so cost scales with company
count and report length, not with how much text is actually relevant --
worth watching as the gold set or max_companies grows.

Costs real API calls. Run: .venv/bin/python eval/eval_grounding.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.llm import ask, estimate_cost
from app.retriever import relevant_companies
from pdf_utils import extract_text_by_page, get_original_page_offset

GOLD_PATH = Path(__file__).parent / "gold_queries.json"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"


def _normalize(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip().lower()


def verify_citation(citation: dict) -> bool:
    """Re-reads the real PDF at the citation's *local* page range (reversing
    the offset correction) and checks cited_text actually appears there."""
    pdf_path = RAW_DIR / f"{citation['document_title']}.pdf"
    if not pdf_path.exists():
        return False

    offset = get_original_page_offset(pdf_path)
    local_start = citation["start_page"] - offset + 1
    local_end = citation["end_page"] - offset + 1

    pages = extract_text_by_page(pdf_path)
    window_text = " ".join(p["text"] for p in pages if local_start <= p["page"] <= local_end)

    return _normalize(citation["cited_text"]) in _normalize(window_text)


def coverage(answer_text: str, citations: list[dict]) -> float:
    """Rough faithfulness proxy: share of answer characters that appear
    inside some citation's cited_text. Not a claim-level faithfulness
    check -- see module docstring."""
    if not answer_text:
        return 0.0
    covered = 0
    for cite in citations:
        # crude but cheap: count cited_text length once per citation,
        # capped at the answer's own length so overlapping citations can't
        # push coverage over 100%
        covered += len(cite["cited_text"])
    return min(1.0, covered / len(answer_text))


def main() -> None:
    gold = json.loads(GOLD_PATH.read_text())["queries"]

    print(f"Grounding eval: {len(gold)} queries, real API calls\n")
    n_citations = n_grounded = 0
    coverages = []
    costs = []

    for item in gold:
        companies = relevant_companies(item["query"], n_results=8, max_companies=4)
        if not companies:
            print(f"[SKIP] {item['query']!r} -- no companies matched, run pipeline/ingest.py first")
            continue

        result = ask(item["query"], companies)
        citations = result["citations"]
        n_citations += len(citations)

        grounded_here = 0
        for c in citations:
            ok = verify_citation(c)
            n_grounded += int(ok)
            grounded_here += int(ok)

        cov = coverage(result["text"], citations)
        coverages.append(cov)

        cost = estimate_cost(result["usage"])
        costs.append(cost)
        usage = result["usage"] or {"input_tokens": 0, "output_tokens": 0}

        print(f"{item['query']!r}")
        print(f"    {len(companies)} companies, {len(citations)} citations, "
              f"{grounded_here} verified grounded, coverage={cov:.0%}")
        print(f"    {usage['input_tokens']:,} input / {usage['output_tokens']:,} output "
              f"tokens, ~${cost:.3f}\n")

    if n_citations == 0:
        print("No citations were returned for any query -- nothing to evaluate. "
              "Make sure pipeline/ingest.py has run against real reports.")
        return

    print(f"Citation grounding: {n_grounded}/{n_citations} ({n_grounded/n_citations:.1%}) "
          f"verified against the actual source PDF")
    print(f"Mean citation coverage of answer text: {sum(coverages)/len(coverages):.1%} "
          f"(faithfulness proxy, not a claim-level check)")
    print(f"Total cost for this run: ${sum(costs):.3f} "
          f"(mean ${sum(costs)/len(costs):.3f}/query, at list price, {len(costs)} queries)")


if __name__ == "__main__":
    main()
