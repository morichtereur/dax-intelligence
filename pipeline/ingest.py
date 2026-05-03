import os
import hashlib
from pathlib import Path
from dotenv import load_dotenv
import chromadb
from pypdf import PdfReader

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

def extract_text_by_page(pdf_path: Path) -> list[dict]:
    reader = PdfReader(str(pdf_path))
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text() or ""
        if text.strip():
            pages.append({"page": i + 1, "text": text})
    return pages

def detect_section(text: str) -> str:
    lower = text.lower()
    for marker in SECTION_MARKERS:
        if marker in lower:
            return marker.replace(" ", "_")
    return "general"

def chunk_text(text: str, chunk_size: int = CHUNK_SIZE, overlap: int = CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks, i = [], 0
    while i < len(words):
        chunks.append(" ".join(words[i:i + chunk_size]))
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

    full_text = "\n".join(p["text"] for p in pages)
    total_pages = pages[-1]["page"]
    chunks = chunk_text(full_text)

    ids, docs, metas = [], [], []
    for i, chunk in enumerate(chunks):
        chunk_id = hashlib.md5(f"{pdf_path.name}_{i}".encode()).hexdigest()
        approx_page = max(1, int((i / len(chunks)) * total_pages))
        section = detect_section(chunk)

        ids.append(chunk_id)
        docs.append(chunk)
        metas.append({
            "company": company,
            "year": year,
            "chunk_index": i,
            "total_chunks": len(chunks),
            "approx_page": approx_page,
            "section": section,
            "source": pdf_path.name
        })

    collection.upsert(documents=docs, ids=ids, metadatas=metas)
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