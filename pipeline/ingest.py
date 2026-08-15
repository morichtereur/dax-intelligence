import os
import hashlib
from pathlib import Path
from dotenv import load_dotenv
import chromadb

from pdf_utils import extract_text_by_page, get_original_page_offset

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"
CHUNK_SIZE = 800
CHUNK_OVERLAP = 100

SECTION_MARKERS = [
    "letter to shareholders", "cfo letter", "management report",
    "risk report", "opportunity report", "consolidated statements",
    "notes to consolidated", "supervisory board report",
    "corporate governance", "sustainability", "segment report",
    "outlook", "guidance", "strategy"
]

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(
    name="dax_reports",
    metadata={"hnsw:space": "cosine"}
)

def detect_section(text: str) -> str:
    lower = text.lower()
    for marker in SECTION_MARKERS:
        if marker in lower:
            return marker.replace(" ", "_")
    return "general"

def chunk_pages(pages: list[dict], chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[dict]:
    """Chunk across pages by word count, same window size as before, but
    keeping each chunk's real page range instead of estimating it later.

    The previous approach joined every page into one string first, chunked
    that, and then guessed a chunk's page from its position alone
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

def ingest_pdf(pdf_path: Path):
    parts = pdf_path.stem.split("_")
    company = parts[0]
    year = parts[-1]

    print(f"Ingesting {pdf_path.name}...")
    pages = extract_text_by_page(pdf_path)

    if not pages:
        print(f"  WARNING: No text extracted — may be scanned/image PDF")
        return

    total_pages = pages[-1]["page"]
    offset = get_original_page_offset(pdf_path)
    chunks = chunk_pages(pages)

    ids, docs, metas = [], [], []
    for i, chunk in enumerate(chunks):
        chunk_id = hashlib.md5(f"{pdf_path.name}_{i}".encode()).hexdigest()
        section = detect_section(chunk["text"])

        ids.append(chunk_id)
        docs.append(chunk["text"])
        metas.append({
            "company": company,
            "year": year,
            "chunk_index": i,
            "total_chunks": len(chunks),
            # real page range of this chunk in the trimmed file, shifted by
            # the original report's page offset -- not an estimate
            "approx_page": chunk["start_page"] + offset - 1,
            "end_page": chunk["end_page"] + offset - 1,
            "section": section,
            "source": pdf_path.name
        })

    collection.upsert(documents=docs, ids=ids, metadatas=metas)
    if offset > 1:
        print(f"  → {len(chunks)} chunks | {total_pages} local pages "
              f"(original report pages {offset}-{offset + total_pages - 1}) | {company} {year}")
    else:
        print(f"  → {len(chunks)} chunks | {total_pages} pages | {company} {year}")

if __name__ == "__main__":
    print(f"Looking for PDFs in: {RAW_DIR.resolve()}\n")
    pdfs = sorted(RAW_DIR.glob("*.pdf"))
    print(f"Found {len(pdfs)} PDFs: {[p.name for p in pdfs]}\n")

    if not pdfs:
        print("No PDFs found — check path above")
    else:
        failed = []
        for pdf in pdfs:
            try:
                ingest_pdf(pdf)
            except Exception as e:
                print(f"  ERROR skipping {pdf.name}: {e}")
                failed.append(pdf.name)
        print(f"\n✓ Done. Total chunks in ChromaDB: {collection.count()}")
        if failed:
            print(f"  Failed: {failed}")
