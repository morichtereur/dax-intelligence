import os
from dotenv import load_dotenv
import chromadb

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(
    name="dax_reports",
    metadata={"hnsw:space": "cosine"}
)

def retrieve(query: str, n_results: int = 8, company_filter: str = None) -> list[dict]:
    where = {"company": company_filter} if company_filter else None
    results = collection.query(
        query_texts=[query],
        n_results=n_results,
        where=where,
        include=["documents", "metadatas", "distances"]
    )
    chunks = []
    for doc, meta, dist in zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    ):
        chunks.append({
            "text": doc,
            "company": meta.get("company"),
            "year": meta.get("year"),
            "section": meta.get("section"),
            "approx_page": meta.get("approx_page"),
            "source": meta.get("source"),
            "score": round(1 - dist, 3)
        })
    return chunks

def get_companies() -> list[str]:
    results = collection.get(include=["metadatas"])
    companies = sorted(set(m["company"] for m in results["metadatas"]))
    return companies