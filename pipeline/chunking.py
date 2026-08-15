"""Text chunking and section tagging. No ChromaDB dependency -- kept
separate from ingest.py (which opens a persistent ChromaDB client as a
module-level side effect) so this logic is importable and unit-testable on
its own, with no external state.
"""

from __future__ import annotations

CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

SECTION_MARKERS = [
    "letter to shareholders", "cfo letter", "management report",
    "risk report", "opportunity report", "consolidated statements",
    "notes to consolidated", "supervisory board report",
    "corporate governance", "sustainability", "segment report",
    "outlook", "guidance", "strategy"
]


def detect_section(text: str) -> str:
    lower = text.lower()
    for marker in SECTION_MARKERS:
        if marker in lower:
            return marker.replace(" ", "_")
    return "general"


def chunk_pages(pages: list[dict], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Chunk across pages by word count, keeping each chunk's real page
    range instead of estimating it from chunk position.

    A prior version joined every page into one string first, chunked that,
    and guessed a chunk's page from its position alone
    (chunk_index / total_chunks * total_pages) -- a linear-interpolation
    estimate that assumes every page has equal text density, and that drifts
    further from the truth the longer the document is. Tracking (word, page)
    pairs from the start makes start_page/end_page exact, not estimated.
    """
    word_pages = [(w, p["page"]) for p in pages for w in p["text"].split()]

    chunks, i = [], 0
    while i < len(word_pages):
        window = word_pages[i:i + chunk_size]
        page_nums = [pg for _, pg in window]
        chunks.append({
            "text": " ".join(w for w, _ in window),
            "start_page": min(page_nums),
            "end_page": max(page_nums),
        })
        i += chunk_size - overlap
    return chunks
