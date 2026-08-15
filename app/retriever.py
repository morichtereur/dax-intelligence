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

def relevant_companies(query: str, n_results: int = 8, company_filter: str = None,
                        max_companies: int = 4) -> list[str]:
    """Which companies' source PDFs should be handed to Claude for this
    query, ranked by their best-matching chunk and deduplicated.

    ChromaDB still does the "which companies are actually relevant"
    narrowing here -- the trimmed section PDFs are small (40-90 pages) but
    attaching all of them to every query would still be wasteful and, past
    a handful of companies, expensive. What changed is what happens after
    this step: the old flow handed the matched *chunks* to Claude and asked
    it to write a page number from memory; now the matched *companies'*
    PDFs go to Claude directly with citations enabled, so page numbers come
    from the API's own parsing of the PDF, not from the model's memory of
    a number a retrieval step attached to a text chunk.
    """
    chunks = retrieve(query, n_results=n_results, company_filter=company_filter)
    companies = []
    for c in chunks:
        if c["company"] not in companies:
            companies.append(c["company"])
    return companies[:max_companies]