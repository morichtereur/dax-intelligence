"""Loads a company's trimmed section PDF(s) as native Claude document
content blocks with citations enabled -- so a citation's page number comes
from the API parsing the actual PDF, not from the model recalling a number
that a retrieval step attached to a text chunk.

Requires the trimmed PDFs pipeline/trim_pdf.py produces to be the files
under data/raw/ (same directory ingest.py reads), named
"{company}_..._{year}.pdf" -- the same convention ingest.py already assumes.
"""

from __future__ import annotations

import base64
from pathlib import Path

RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

# Claude API limit: 600 pages per PDF document (100 on 200k-context models
# like Haiku). Trimmed section PDFs should be well under this, but check
# rather than let the API reject the request with no context on why.
MAX_PDF_PAGES = 600


def load_company_documents(companies: list[str]) -> tuple[list[dict], dict[str, int]]:
    """Returns (document content blocks, {title: original_page_offset}).

    The offset matters even here: the Citations API returns page numbers
    local to the attached PDF (the trimmed section file), not the original
    report -- same problem pipeline/ingest.py solves for the ChromaDB path,
    solved the same way, by reading the offset pipeline/trim_pdf.py stamped
    into the trimmed file's own metadata. Caller (app/llm.py) is expected to
    add (offset - 1) to whatever page numbers the API returns for a given
    document_title.
    """
    from pypdf import PdfReader  # local import: keeps this module's other
    # dependency (base64/Path) importable without pypdf installed, e.g. from
    # a quick script that just wants RAW_DIR
    from pipeline.pdf_utils import get_original_page_offset

    blocks = []
    offsets = {}
    for company in companies:
        for pdf_path in sorted(RAW_DIR.glob(f"{company}_*.pdf")):
            n_pages = len(PdfReader(str(pdf_path)).pages)
            if n_pages > MAX_PDF_PAGES:
                print(f"  WARNING: {pdf_path.name} has {n_pages} pages, over the "
                      f"{MAX_PDF_PAGES}-page API limit -- trim it further, skipping for now")
                continue

            title = pdf_path.stem
            offsets[title] = get_original_page_offset(pdf_path)
            data = base64.standard_b64encode(pdf_path.read_bytes()).decode("utf-8")
            blocks.append({
                "type": "document",
                "source": {"type": "base64", "media_type": "application/pdf", "data": data},
                "title": title,
                "citations": {"enabled": True},
            })
    return blocks, offsets
