"""PDF slicing and page-accounting helpers. No ChromaDB dependency -- kept
separate from ingest.py so trim_pdf.py can use extract_section() without
opening a persistent ChromaDB client just to cut a PDF.
"""

from __future__ import annotations

from pathlib import Path

from pypdf import PdfReader, PdfWriter

# Custom metadata key written by extract_section() and read back by
# get_original_page_offset(). Lets a trimmed section PDF (e.g. "just the
# Lagebericht, pages 20-100 of the original 400-page report") still produce
# page citations against the real, publicly filed report -- not against
# page 1 of the trimmed file.
ORIGINAL_PAGE_START_KEY = "/OriginalPageStart"


def extract_section(pdf_path: Path, start_page: int, end_page: int, output_path: Path) -> None:
    """Slice pages [start_page, end_page] (1-indexed, inclusive, as printed
    in the report's own table of contents) out of pdf_path into a new PDF at
    output_path.

    Pages are copied as real page objects (writer.add_page), not
    re-extracted or re-flowed text -- the trimmed file reads identically to
    the corresponding pages of the original filing.

    Also relevant to the 600-page (100-page on 200k-context models) PDF
    limit on the LLM API: a section-trimmed file this way typically lands
    at 40-90 pages, comfortably under either cap, without an arbitrary
    page-count cutoff that could split a report mid-section.

    start_page is written into the output PDF's own metadata
    (ORIGINAL_PAGE_START_KEY) so ingest_pdf() can recover it later and
    compute page citations against the original report's real pagination,
    not the trimmed file's.
    """
    reader = PdfReader(str(pdf_path))
    writer = PdfWriter()

    if not (1 <= start_page <= end_page <= len(reader.pages)):
        raise ValueError(
            f"start_page/end_page ({start_page}-{end_page}) out of range for "
            f"{pdf_path.name} ({len(reader.pages)} pages)"
        )

    for page in reader.pages[start_page - 1:end_page]:
        writer.add_page(page)

    writer.add_metadata({ORIGINAL_PAGE_START_KEY: str(start_page)})

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "wb") as f:
        writer.write(f)

    print(f"  Extracted pages {start_page}-{end_page} of {pdf_path.name} "
          f"({end_page - start_page + 1} pages) -> {output_path.name}")


def extract_text_by_page(pdf_path: Path) -> list[dict]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page": i + 1, "text": text})
    return pages


def get_original_page_offset(pdf_path: Path) -> int:
    """1 for a full/untrimmed report; the recorded start page for a file
    produced by extract_section()."""
    reader = PdfReader(str(pdf_path))
    meta = reader.metadata or {}
    offset = meta.get(ORIGINAL_PAGE_START_KEY)
    return int(offset) if offset else 1
