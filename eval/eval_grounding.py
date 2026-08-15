"""Generation eval: citation grounding + faithfulness, for
eval/gold_queries.json run through app/retriever.py's retrieve() and
app/llm.py's ask().

Unlike a citations-API architecture where a citation's *location* can't be
hallucinated (the API extracts it from the attached PDF), this app has
Claude write (Company[, YYYY], p.N) into free-text prose from a set of
retrieved excerpts, so the location itself can be wrong — that's exactly
what app/llm.py's verify_citations() ground-truth checks (does that
company/year/page combination actually appear among the retrieved chunks),
and what this eval reports in aggregate across the gold set.

Two faithfulness signals, cheapest first:

1. Citation grounding rate: what share of citations verify_citations()
   marks verified. Free (no extra API call) — this is this project's core
   anti-hallucination mechanism, so its own pass rate is the first thing
   worth tracking over time.
2. Claim-level faithfulness: a judge model call (judge_faithfulness) that
   splits the answer into its distinct factual claims and checks each one
   against the retrieved excerpts actually used, the same way a human
   reviewer would — not by counting characters. This catches a claim that
   carries a *verified* citation but still overstates or misreads what the
   cited excerpt actually says, which grounding rate alone can't see.

Also reports token usage and $ cost per query (app/llm.py's estimate_cost).

Costs two real API calls per query -- generation (MODEL) plus judge
(JUDGE_MODEL, a cheaper model since it's classification, not generation).
Run: .venv/bin/python eval/eval_grounding.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from app.llm import ask, client, estimate_cost, verify_citations
from app.retriever import retrieve, relevant_companies

GOLD_PATH = Path(__file__).parent / "gold_queries.json"

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


def judge_faithfulness(answer_text: str, chunks: list[dict]) -> dict:
    """Claim-level faithfulness check: a judge model call that splits the
    answer into its distinct factual claims and checks each one against the
    retrieved excerpts, instead of trusting the inline citations alone.
    tool_choice forces structured output so results parse reliably instead
    of scraping free text.

    Returns {"claims": [...], "score": supported/total or None if the
    answer had no extractable claims, "usage": {...}}."""
    if not answer_text.strip():
        return {"claims": [], "score": None, "usage": None}

    evidence = "\n".join(
        f"- [{c['company']} {c['year']}, p.{c['approx_page']}] {c['text'][:500]}" for c in chunks
    ) or "(no excerpts were retrieved for this query)"

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
    faithfulness_scores = []
    costs = []

    for item in gold:
        chunks = retrieve(item["query"], n_results=8)
        if not chunks:
            print(f"[SKIP] {item['query']!r} -- nothing retrieved, run pipeline/ingest.py first")
            continue

        result = ask(item["query"], chunks)
        answer, usage = result["text"], result["usage"]

        citations = verify_citations(answer, chunks)
        n_citations += len(citations)
        n_grounded_here = sum(1 for c in citations if c["verified"])
        n_grounded += n_grounded_here

        judged = judge_faithfulness(answer, chunks)
        unsupported = [c for c in judged["claims"] if not c["supported"]]
        if judged["score"] is not None:
            faithfulness_scores.append(judged["score"])

        gen_cost = estimate_cost(usage)
        judge_cost = estimate_cost(judged["usage"], model=JUDGE_MODEL)
        costs.append(gen_cost + judge_cost)
        usage = usage or {"input_tokens": 0, "output_tokens": 0}

        companies = sorted({c["company"] for c in chunks})
        print(f"{item['query']!r}")
        print(f"    companies retrieved: {companies}")
        print(f"    {len(citations)} citations, {n_grounded_here} verified grounded")
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
          f"verified against retrieved excerpts")
    if faithfulness_scores:
        print(f"Mean claim-level faithfulness: {sum(faithfulness_scores)/len(faithfulness_scores):.1%} "
              f"({len(faithfulness_scores)} queries judged by {JUDGE_MODEL})")
    print(f"Total cost for this run: ${sum(costs):.3f} "
          f"(mean ${sum(costs)/len(costs):.3f}/query incl. judge, at list price, {len(costs)} queries)")


if __name__ == "__main__":
    main()
