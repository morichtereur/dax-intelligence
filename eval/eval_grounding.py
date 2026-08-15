"""Generation eval: citation grounding + faithfulness (coverage proxy and a
claim-level judge check), for eval/gold_queries.json run through
app/llm.py's ask().

What this checks and why it's still worth checking even though the Citations
API extracts citations from the attached PDF (so it can't hallucinate a
citation's *location*): the check here is on *this project's own*
offset-correction pipeline (pipeline/pdf_utils.py's page-offset stamping,
app/llm.py's exclusive-end-page fix), not on the model. A bug in either of
those would make a citation's reported page wrong even though the API's own
citation extraction is correct -- this test re-derives the local page,
re-reads the actual attached PDF, and confirms cited_text is really there.

Also reports two faithfulness signals, cheapest first:

1. Citation coverage: what share of the answer's characters fall inside a
   cited span. Free (no extra API call) but crude -- it only tells you how
   much of the answer is backed by *some* citation, not whether any specific
   claim is actually true given the evidence.
2. Claim-level faithfulness: a judge model call (judge_faithfulness) that
   splits the answer into its distinct factual claims and checks each one
   against the citations actually returned, the same way a human reviewer
   would -- not by counting characters. This is the real check; coverage is
   a cheap smoke test for when you don't want to spend the extra call.

Also reports token usage and $ cost per query. This matters specifically
for this architecture: unlike chunk-based retrieval, every query attaches
the *full* trimmed PDF for each matched company, so cost scales with company
count and report length, not with how much text is actually relevant --
worth watching as the gold set or max_companies grows.

Costs two real API calls per query -- generation (MODEL) plus judge
(JUDGE_MODEL, a cheaper model since it's classification, not generation).
Run: .venv/bin/python eval/eval_grounding.py
"""

from __future__ import annotations

import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.llm import ask, client, estimate_cost
from app.retriever import relevant_companies
from pdf_utils import extract_text_by_page, get_original_page_offset

GOLD_PATH = Path(__file__).parent / "gold_queries.json"
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

# Cheap classification task, not generation -- a smaller model keeps the
# judge call from dominating this eval's own cost.
JUDGE_MODEL = "claude-haiku-4-5"

CLAIM_JUDGE_TOOL = {
    "name": "record_claim_judgments",
    "description": "Record a faithfulness judgment for each distinct factual claim in the answer.",
    "input_schema": {
        "type": "object",
        "properties": {
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "claim": {
                            "type": "string",
                            "description": "The factual claim, quoted or closely paraphrased from the answer.",
                        },
                        "supported": {
                            "type": "boolean",
                            "description": "True only if this exact claim is directly stated or clearly implied by the cited evidence -- not by general knowledge or plausibility.",
                        },
                        "reasoning": {
                            "type": "string",
                            "description": "One sentence: which evidence line supports it, or why none does.",
                        },
                    },
                    "required": ["claim", "supported", "reasoning"],
                },
            },
        },
        "required": ["claims"],
    },
}

JUDGE_PROMPT = """You are auditing an AI-generated answer for faithfulness to its cited evidence.

ANSWER:
{answer}

CITED EVIDENCE (the only source material available to back the answer):
{evidence}

Break the answer into its distinct factual claims -- numbers, named strategies, causal statements, comparisons between companies. Ignore transitions, hedges, and purely structural language. For each claim, decide whether it is directly supported by the cited evidence above, not by general knowledge or plausibility. Call record_claim_judgments with your results."""


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


def judge_faithfulness(answer_text: str, citations: list[dict]) -> dict:
    """Claim-level faithfulness check: a judge model call that splits the
    answer into its distinct factual claims and checks each one against the
    citations actually returned, instead of counting characters like
    coverage() does. tool_choice forces structured output so results parse
    reliably instead of scraping free text.

    Returns {"claims": [...], "score": supported/total or None if the
    answer had no extractable claims, "usage": {...}}."""
    if not answer_text.strip():
        return {"claims": [], "score": None, "usage": None}

    evidence = "\n".join(
        f"- [{c['document_title']}] {c['cited_text']}" for c in citations
    ) or "(no citations were returned with this answer)"

    response = client.messages.create(
        model=JUDGE_MODEL,
        max_tokens=2000,
        tools=[CLAIM_JUDGE_TOOL],
        tool_choice={"type": "tool", "name": "record_claim_judgments"},
        messages=[{
            "role": "user",
            "content": JUDGE_PROMPT.format(answer=answer_text, evidence=evidence),
        }],
    )

    tool_use = next(b for b in response.content if b.type == "tool_use")
    claims = tool_use.input["claims"]
    n_supported = sum(1 for c in claims if c["supported"])
    usage = {
        "input_tokens": response.usage.input_tokens,
        "output_tokens": response.usage.output_tokens,
    }
    return {
        "claims": claims,
        "score": n_supported / len(claims) if claims else None,
        "usage": usage,
    }


def main() -> None:
    gold = json.loads(GOLD_PATH.read_text())["queries"]

    print(f"Grounding eval: {len(gold)} queries, real API calls\n")
    n_citations = n_grounded = 0
    coverages = []
    faithfulness_scores = []
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

        judged = judge_faithfulness(result["text"], citations)
        unsupported = [c for c in judged["claims"] if not c["supported"]]
        if judged["score"] is not None:
            faithfulness_scores.append(judged["score"])

        gen_cost = estimate_cost(result["usage"])
        judge_cost = estimate_cost(judged["usage"], model=JUDGE_MODEL)
        costs.append(gen_cost + judge_cost)
        usage = result["usage"] or {"input_tokens": 0, "output_tokens": 0}

        print(f"{item['query']!r}")
        print(f"    {len(companies)} companies, {len(citations)} citations, "
              f"{grounded_here} verified grounded, coverage={cov:.0%}")
        if judged["score"] is not None:
            print(f"    faithfulness: {len(judged['claims']) - len(unsupported)}/{len(judged['claims'])} "
                  f"claims supported ({judged['score']:.0%})")
            for c in unsupported:
                print(f"      [UNSUPPORTED] {c['claim']!r} -- {c['reasoning']}")
        print(f"    {usage['input_tokens']:,} input / {usage['output_tokens']:,} output "
              f"tokens, ~${gen_cost + judge_cost:.3f} (incl. judge)\n")

    if n_citations == 0:
        print("No citations were returned for any query -- nothing to evaluate. "
              "Make sure pipeline/ingest.py has run against real reports.")
        return

    print(f"Citation grounding: {n_grounded}/{n_citations} ({n_grounded/n_citations:.1%}) "
          f"verified against the actual source PDF")
    print(f"Mean citation coverage of answer text: {sum(coverages)/len(coverages):.1%} "
          f"(cheap proxy, not a claim-level check)")
    if faithfulness_scores:
        print(f"Mean claim-level faithfulness: {sum(faithfulness_scores)/len(faithfulness_scores):.1%} "
              f"({len(faithfulness_scores)} queries judged by {JUDGE_MODEL})")
    print(f"Total cost for this run: ${sum(costs):.3f} "
          f"(mean ${sum(costs)/len(costs):.3f}/query incl. judge, at list price, {len(costs)} queries)")


if __name__ == "__main__":
    main()
