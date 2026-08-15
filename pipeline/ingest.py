import os
import hashlib
from pathlib import Path
from dotenv import load_dotenv
import chromadb

from pdf_utils import extract_text_by_page, get_original_page_offset
from chunking import detect_section, chunk_pages

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
RAW_DIR = Path(__file__).parent.parent / "data" / "raw"

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(
    name="dax_reports",
    metadata={"hnsw:space": "cosine"}
)

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
