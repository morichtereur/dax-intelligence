# DAX Intelligence 📊

AI-powered analysis of DAX 40 annual reports — built for finance and strategy professionals.

Ask questions across 15 DAX company reports (FY2025) and get answers with source citations, page references, and cross-company comparisons.

## What it does

- Query 15 DAX annual reports simultaneously
- Get answers with company + page citations
- Compare how companies frame the same topic differently
- Filter by company or query across all at once

**Example queries:**
- "How do Siemens and SAP frame their AI investment strategy?"
- "Which companies disclosed structural cost reduction programs?"
- "What do CFOs say about macroeconomic risks in 2025?"
- "Compare how BMW and Mercedes frame EV transition costs"

## Architecture

PDF Reports → pypdf extraction → word-based chunking (800w, 100w overlap) → section detection (CFO letter, outlook, risk report, etc.) → ChromaDB cosine similarity → Claude Sonnet synthesis

Two-query design: ChromaDB retrieves the most relevant chunks, Claude synthesises them with a finance analyst system prompt that enforces citations and consulting-grade framing.

## Companies covered (FY2025)

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

## Stack

- **Retrieval:** ChromaDB (persistent, cosine similarity)
- **Embeddings:** ChromaDB default (all-MiniLM-L6-v2)
- **LLM:** Claude Sonnet via Anthropic API
- **Frontend:** Streamlit
- **PDF parsing:** pypdf

## Design decisions

Word-based chunking (800 words, 100 overlap) over character-based — more stable across the mixed layouts of German annual reports. Section detection is heuristic (keyword matching on common annual report headers) rather than ML-based, which is fast and sufficient for this retrieval use case. ChromaDB chosen over FAISS for its persistent client and metadata filtering, useful for company-level scoping without re-embedding.

## Requirements

    anthropic
    chromadb
    streamlit
    pypdf
    python-dotenv
    pycryptodome

---

Built by [Moritz Richter](https://www.linkedin.com/in/morichtereur) · Finance & Strategy Consultant · Zürich
