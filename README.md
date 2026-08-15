# DAX Intelligence 📊

[![Tests](https://github.com/morichtereur/dax-intelligence/actions/workflows/test.yml/badge.svg)](https://github.com/morichtereur/dax-intelligence/actions/workflows/test.yml)

AI-powered analysis of DAX 40 annual reports — built for finance and strategy professionals.

Ask questions across DAX company reports and get answers with citations
whose page numbers come from Claude's own PDF parsing, not from a retrieval
step's guess — plus a retrieval + generation eval harness so "citation
accuracy" is a checked number, not a claim.

## What it does

- Query multiple DAX annual reports simultaneously
- Get answers with company + page citations, extracted by the API directly
  from the source PDF (not written into prose from memory)
- Compare how companies frame the same topic differently
- Filter by company or query across all at once

**Example queries:**
- "How do Siemens and SAP frame their AI investment strategy?"
- "What do CFOs say about macroeconomic risks in 2025?"
- "Compare how BMW and Mercedes frame EV transition costs"

## Architecture

    Full annual report PDF (150-600+ pages)
            ↓  pipeline/trim_pdf.py — manual, once per company
    Section-trimmed PDF (Letter to Shareholders → Management Report →
    Risk Report → Outlook; ~40-90 pages, real page offset stamped in metadata)
            ↓  pipeline/ingest.py — pypdf extraction, page-aware chunking
    ChromaDB (cosine similarity) — ranks which companies are relevant to a query
            ↓  app/documents.py — loads the *whole* trimmed PDF per relevant company
    Claude, with citations: enabled on each document — answer + real
    page-level citations, corrected back to the original report's pagination

The two-stage design matters: ChromaDB decides *which companies'* reports are
relevant (cheap, scales past a handful of companies); Claude then reads the
*actual PDF* for those companies rather than a handful of retrieved text
chunks, so page citations are extracted from the document itself instead of
recalled by the model from a number a retrieval step attached to a chunk.

**Why the trimming step exists:** the Claude API caps PDF input at 600 pages
(100 on 200k-context models), and most DAX annual reports run 300-600+
pages, mostly Notes to Consolidated Statements and governance boilerplate
that isn't relevant to strategic-narrative questions anyway. Trimming to the
narrative sections (a) fits under the API limit without an arbitrary
mid-section cutoff, and (b) removes the noise a semantic retrieval step
would otherwise have to filter around.

## Companies covered

BMW · Mercedes-Benz · Volkswagen · Siemens · Siemens Energy · Allianz ·
Munich Re · SAP · Infineon · BASF (FY2025) — curated for natural
cross-company comparisons (auto, industrials, insurance) rather than
exhaustive DAX 40 coverage; see `eval/gold_queries.json` for the reasoning.

## Quickstart

    git clone https://github.com/morichtereur/dax-intelligence
    cd dax-intelligence
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env

    # 1. For each company: find the Letter to Shareholders / Management
    #    Report / Outlook page range in the report's own table of contents,
    #    then trim to it:
    python3 pipeline/trim_pdf.py path/to/BMW_full_report_2025.pdf 20 100 \
        data/raw/BMW_management_report_2025.pdf

    # 2. Ingest the trimmed PDFs
    python3 pipeline/ingest.py

    # 3. Run it
    streamlit run app/app.py

## Testing

    pip install -r requirements-dev.txt
    pytest tests/ -v

Covers the deterministic core (PDF page slicing, offset stamping/recovery,
page-aware chunking) with synthetic PDF fixtures — no real reports or API
key needed, which is also what runs in CI on every push.

## Eval

Two scripts, run once real reports are ingested (they hit the live
ChromaDB collection and, for grounding, the real API):

    python3 eval/eval_retrieval.py   # precision@k / recall@k against eval/gold_queries.json
    python3 eval/eval_grounding.py   # citation grounding + a coverage/faithfulness proxy — costs real API calls

`eval_retrieval.py` checks whether `relevant_companies()` finds the right
company for a query. Every gold query names its expected company explicitly
in the question — that's what makes "expected" an objective label rather
than a judgment call; a topic query like "which companies discussed cost
cuts" can't be graded this way without reading every report by hand.

`eval_grounding.py` re-derives each citation's real page, re-reads the
actual attached PDF at that page, and confirms the cited text is really
there — checking this project's own offset-correction pipeline, not
whether Claude hallucinates (it structurally can't invent a citation's
*location*, since the API extracts citations from the attached document).
It also reports citation coverage of the answer text as a rough
faithfulness proxy — how much of the answer is backed by a citation at all,
not a claim-by-claim faithfulness check.

## Stack

- **Retrieval:** ChromaDB (persistent, cosine similarity) — company-level relevance ranking only
- **Generation:** Claude, with native PDF citations (`citations: {enabled: true}`) on each attached report
- **PDF handling:** pypdf — section trimming, page-aware chunking, offset metadata
- **Frontend:** Streamlit
- **Testing:** pytest (synthetic-PDF unit tests, run in CI) + a manual eval harness (retrieval precision/recall, citation grounding) for real data
- **CI:** GitHub Actions

## Design decisions

Word-based chunking (800 words, 100 overlap), tracking each chunk's real
page range from the source PDF's own page boundaries rather than estimating
it from chunk position — the earlier linear-interpolation approach assumed
uniform text density per page, which drifts further from the truth the
longer a document is. Section detection is heuristic (keyword matching on
common annual report headers), used at ingest time to help decide what to
trim; the trimmed PDF itself, not chunk-level section metadata, is what
Claude reads at generation time. ChromaDB chosen over FAISS for its
persistent client and metadata filtering, useful for company-level scoping
without re-embedding.

---

Built by [Moritz Richter](https://www.linkedin.com/in/moritz-richter-28297119a) · Finance & Strategy Consultant · Zürich
