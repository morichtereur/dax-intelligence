# DAX Intelligence 📊

[![Tests](https://github.com/morichtereur/dax-intelligence/actions/workflows/test.yml/badge.svg)](https://github.com/morichtereur/dax-intelligence/actions/workflows/test.yml)

AI-powered analysis of DAX 40 annual reports — built for finance and strategy professionals.

Ask questions across 15 DAX company reports (FY2024 & FY2025) and get a research memo with source citations, an audit trail linking straight back to the exact page of the original filing, and year-over-year framing when both fiscal years are in scope.

## What it does

- Query 15 DAX annual reports across two fiscal years simultaneously
- Get answers with company + page citations, checked against the retrieved excerpts before they're shown — not just asserted
- Compare how companies frame the same topic differently, or how one company's own framing shifted year over year
- Filter by company, fiscal year, or query across everything at once
- Click a citation's page number to open the source PDF at that exact page
- Export any answer as a print-ready PDF memo

**Example queries:**
- "How do Siemens and SAP frame their AI investment strategy?"
- "Which companies disclosed structural cost reduction programs?"
- "What do CFOs say about macroeconomic risks in 2025?"
- "How did Munich Re's view on geopolitical risk change between 2024 and 2025?"

## Architecture

    PDF Reports → pypdf extraction → page-tracked word chunking (800w, 100w
    overlap; every chunk keeps the real page range its text came from, not
    an estimate) → section detection → ChromaDB (cosine similarity)
            ↓
    Hybrid retrieval: ChromaDB semantic search + BM25 keyword search, their
    results unioned rather than one re-scoring the other, then a
    cross-encoder (ms-marco-MiniLM-L-6-v2) re-ranks the combined candidate
    pool on actual query/passage relevance
            ↓
    Claude Sonnet synthesis — cites (Company[, Year], p.N) inline; every
    citation is regex-parsed back out and checked against the chunks the
    model actually received before the answer is shown

Retrieval confidence bands (calibrated against this corpus: on-topic
queries cluster ~0.55–0.60 cosine similarity, off-topic ones sit ~0.19–0.26)
gate the whole pipeline — a query with no real answer in the corpus never
reaches the model at all, and a weak-but-present match is shown with an
explicit low-confidence banner instead of being presented as solid ground.

**Why not send whole PDFs with Claude's native citations API instead?**
That approach (extracting citations directly from an attached PDF rather
than verifying a model-written page number) is genuinely more precise per
citation, and worth using — see `pipeline/trim_pdf.py` and the git history
for that direction. It requires manually trimming each report to its
narrative section first (the API caps PDF input at 600 pages, and most
annual reports run 300–600+), which doesn't scale to 15 companies × 2
fiscal years without a lot of by-hand work. This project optimizes for
broad, low-effort coverage instead: drop a PDF in `data/raw/`, run
`pipeline/ingest.py`, done — and closes most of the resulting gap with
real page-range tracking (not a linear estimate) plus ground-truth citation
verification against the retrieved excerpts.

## Companies covered (FY2024 & FY2025)

Allianz · BASF · Bayer · Beiersdorf · BMW · DHL · Henkel · Infineon · Mercedes-Benz · Merck KGaA · Munich Re · SAP · Siemens · Siemens Energy · Volkswagen

## Quickstart

    git clone https://github.com/morichtereur/dax-intelligence
    cd dax-intelligence
    python3 -m venv .venv && source .venv/bin/activate
    pip install -r requirements.txt
    cp .env.example .env
    # Add PDF reports to data/raw/ named Company_Report_YYYY.pdf
    python3 pipeline/ingest.py
    streamlit run app/app.py

## Eval

    pip install -r requirements-dev.txt
    pytest tests/ -v                       # chunking + page-offset unit tests, no API calls
    .venv/bin/python eval/eval_retrieval.py # precision/recall of company retrieval against eval/gold_queries.json
    .venv/bin/python eval/eval_grounding.py # citation grounding rate + a claim-level LLM faithfulness judge, real API calls

## Stack

- **Retrieval:** ChromaDB (persistent, cosine similarity) + BM25 (`rank-bm25`) + cross-encoder re-ranking (`sentence-transformers`)
- **Embeddings:** ChromaDB default (all-MiniLM-L6-v2)
- **LLM:** Claude Sonnet via Anthropic API
- **Frontend:** Streamlit
- **PDF parsing/export:** pypdf, reportlab

## Design decisions

Word-based chunking (800 words, 100 overlap) over character-based — more stable across the mixed layouts of German annual reports. Each chunk tracks the real page range its words came from (`pipeline/chunking.py`), not a chunk-index/total-chunks linear estimate, which drifts further from the truth the longer and more unevenly-formatted the report. Section detection is heuristic (keyword matching on common annual report headers) rather than ML-based, which is fast and sufficient for this retrieval use case. ChromaDB chosen over FAISS for its persistent client and metadata filtering, useful for company- and year-level scoping without re-embedding.

## Requirements

    anthropic
    chromadb
    streamlit
    pypdf
    python-dotenv
    pycryptodome
    rank-bm25
    sentence-transformers
    reportlab
    watchdog

---

Built by [Moritz Richter](https://www.linkedin.com/in/morichtereur) · Finance & Strategy Consultant · Zürich
