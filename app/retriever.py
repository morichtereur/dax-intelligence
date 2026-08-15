import os
from dotenv import load_dotenv
import chromadb
from rank_bm25 import BM25Okapi
from sentence_transformers import CrossEncoder

load_dotenv()

CHROMA_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")

# Cosine-similarity bands calibrated against this corpus (all-MiniLM-L6-v2):
# on-topic queries cluster ~0.55-0.60, off-topic/out-of-scope queries sit
# ~0.19-0.26, and a dead zone in between (~0.28-0.45) holds chunks that share
# vocabulary but may not actually support an answer. MIN_RELEVANCE discards
# the former as noise; LOW_CONFIDENCE flags the latter for the UI instead of
# presenting it as solid ground. Both are calibrated on the semantic-search
# score alone, so the confidence UI keeps meaning even though retrieve()
# below also folds in a keyword pass and a re-ranker.
MIN_RELEVANCE = 0.28
LOW_CONFIDENCE = 0.45

RERANK_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"

client = chromadb.PersistentClient(path=CHROMA_DIR)
collection = client.get_or_create_collection(
    name="dax_reports",
    metadata={"hnsw:space": "cosine"}
)

_cross_encoder: CrossEncoder | None = None
_bm25_cache: dict[str, tuple[BM25Okapi, list[dict]]] = {}


def _get_cross_encoder() -> CrossEncoder:
    global _cross_encoder
    if _cross_encoder is None:
        _cross_encoder = CrossEncoder(RERANK_MODEL)
    return _cross_encoder


def _where(company_filter: str | None, year_filter: str | None) -> dict | None:
    clauses = []
    if company_filter:
        clauses.append({"company": company_filter})
    if year_filter:
        clauses.append({"year": year_filter})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


def _get_bm25_index(company_filter: str | None, year_filter: str | None) -> tuple[BM25Okapi, list[dict]]:
    """BM25 catches the exact numbers, tickers and section headers that
    embeddings are weak on ("EBIT margin", "p.68") — built once per scope
    (company/year combination) and cached for the process lifetime, since
    the underlying report set doesn't change during a session."""
    key = f"{company_filter or '__all__'}::{year_filter or '__all__'}"
    if key not in _bm25_cache:
        where = _where(company_filter, year_filter)
        data = collection.get(where=where, include=["documents", "metadatas"])
        chunk_list = [
            {"id": _id, "text": doc, "meta": meta}
            for _id, doc, meta in zip(data["ids"], data["documents"], data["metadatas"])
        ]
        tokenized = [c["text"].lower().split() for c in chunk_list]
        _bm25_cache[key] = (BM25Okapi(tokenized), chunk_list)
    return _bm25_cache[key]


def _to_chunk(id_: str, text: str, meta: dict, score: float | None, via: str) -> dict:
    return {
        "id": id_,
        "text": text,
        "company": meta.get("company"),
        "year": meta.get("year"),
        "section": meta.get("section"),
        "approx_page": meta.get("approx_page"),
        "source": meta.get("source"),
        "score": score,
        "via": via,
    }


def retrieve(query: str, n_results: int = 8, company_filter: str = None, year_filter: str = None) -> list[dict]:
    where = _where(company_filter, year_filter)
    fetch_n = min(max(n_results * 4, 32), collection.count() or 1)

    # 1) Semantic pass — also the one that drives the confidence bands above.
    results = collection.query(
        query_texts=[query],
        n_results=fetch_n,
        where=where,
        include=["documents", "metadatas", "distances"]
    )
    pool: dict[str, dict] = {}
    for id_, doc, meta, dist in zip(
        results["ids"][0], results["documents"][0], results["metadatas"][0], results["distances"][0]
    ):
        score = round(1 - dist, 3)
        if score < MIN_RELEVANCE:
            continue
        pool[id_] = _to_chunk(id_, doc, meta, score, "embedding")

    if not pool:
        return []

    # 2) Keyword pass — union in, don't just re-score the same candidates,
    # so a chunk embeddings ranked outside fetch_n but that contains the
    # exact term/number asked about still gets a chance to surface.
    bm25, chunk_list = _get_bm25_index(company_filter, year_filter)
    bm25_scores = bm25.get_scores(query.lower().split())
    top_bm25 = sorted(range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True)[:fetch_n]
    for i in top_bm25:
        if bm25_scores[i] <= 0:
            continue
        c = chunk_list[i]
        if c["id"] in pool:
            pool[c["id"]]["via"] = "both"
        else:
            pool[c["id"]] = _to_chunk(c["id"], c["text"], c["meta"], None, "keyword")

    candidates = list(pool.values())

    # 3) Cross-encoder re-ranks the union pool on actual query/passage
    # relevance rather than vector geometry or term overlap alone.
    if len(candidates) > n_results:
        ce = _get_cross_encoder()
        pairs = [(query, c["text"]) for c in candidates]
        for c, s in zip(candidates, ce.predict(pairs)):
            c["_rerank"] = float(s)
        candidates.sort(key=lambda c: c["_rerank"], reverse=True)

    final = candidates[:n_results]
    final.sort(key=lambda c: c["score"] if c["score"] is not None else -1, reverse=True)
    for c in final:
        c.pop("id", None)
        c.pop("_rerank", None)
    return final


def get_companies() -> list[str]:
    results = collection.get(include=["metadatas"])
    companies = sorted(set(m["company"] for m in results["metadatas"]))
    return companies


def get_years() -> list[str]:
    results = collection.get(include=["metadatas"])
    years = sorted(set(m["year"] for m in results["metadatas"] if m.get("year")))
    return years
