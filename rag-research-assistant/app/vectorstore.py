"""
FAISS-backed vector store (swapped in place of ChromaDB to avoid native
build/compilation issues on Windows — faiss-cpu ships solid prebuilt
wheels and needs no separate build toolchain or server process).

Embeddings run locally via sentence-transformers (Groq does not currently
expose an embeddings endpoint, so pairing a fast open embedding model with
Groq's ultra-fast LPU inference for generation gives the best of both:
free/local embedding + extremely low-latency answer generation).

The FAISS index is kept as an in-process singleton and persisted to disk
(via save_local/load_local) after every write so the corpus survives
restarts.
"""
from functools import lru_cache
from pathlib import Path
from threading import Lock
from typing import List, Optional, Callable

import faiss
from langchain_community.vectorstores import FAISS
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

from app import config

_write_lock = Lock()  # guards add/delete + save_local so concurrent API calls don't corrupt the index


@lru_cache(maxsize=1)
def get_embeddings() -> HuggingFaceEmbeddings:
    return HuggingFaceEmbeddings(model_name=config.EMBEDDING_MODEL)


@lru_cache(maxsize=1)
def get_vectorstore() -> FAISS:
    embeddings = get_embeddings()
    index_dir = Path(config.FAISS_INDEX_DIR)
    index_file = index_dir / "index.faiss"

    if index_file.exists():
        return FAISS.load_local(
            str(index_dir),
            embeddings,
            allow_dangerous_deserialization=True,  # index was created by us, trusted
        )

    # No persisted index yet: create an empty one sized to the embedding dimension.
    dim = len(embeddings.embed_query("dimension probe"))
    index = faiss.IndexFlatL2(dim)
    return FAISS(
        embedding_function=embeddings,
        index=index,
        docstore=InMemoryDocstore(),
        index_to_docstore_id={},
    )


def _persist(vs: FAISS) -> None:
    Path(config.FAISS_INDEX_DIR).mkdir(parents=True, exist_ok=True)
    vs.save_local(config.FAISS_INDEX_DIR)


def add_documents(documents: List[Document]) -> List[str]:
    vs = get_vectorstore()
    ids = [d.metadata["chunk_id"] for d in documents]
    with _write_lock:
        vs.add_documents(documents=documents, ids=ids)
        _persist(vs)
    return ids


def delete_paper(arxiv_id: str) -> int:
    """Delete all chunks belonging to a given paper."""
    vs = get_vectorstore()
    with _write_lock:
        ids = [
            doc_id
            for doc_id, doc in vs.docstore._dict.items()
            if doc.metadata.get("arxiv_id") == arxiv_id
        ]
        if ids:
            vs.delete(ids=ids)
            _persist(vs)
    return len(ids)


def get_all_documents(arxiv_ids: Optional[List[str]] = None) -> List[Document]:
    """Fetch all stored chunks (optionally filtered to specific papers) —
    used to (re)build the in-memory BM25 index and for corpus listings."""
    vs = get_vectorstore()
    docs = list(vs.docstore._dict.values())
    if arxiv_ids:
        allowed = set(arxiv_ids)
        docs = [d for d in docs if d.metadata.get("arxiv_id") in allowed]
    return docs


def metadata_filter(arxiv_ids: Optional[List[str]] = None) -> Optional[Callable[[dict], bool]]:
    """Build a FAISS-compatible metadata filter (callable form) restricting
    results to the given arXiv IDs, or None for no restriction."""
    if not arxiv_ids:
        return None
    allowed = set(arxiv_ids)
    return lambda meta: meta.get("arxiv_id") in allowed


def list_papers() -> dict:
    """Return {arxiv_id: {title, sections, num_chunks, filename}} for every
    ingested paper currently in the store."""
    docs = get_all_documents()
    papers = {}
    for d in docs:
        aid = d.metadata["arxiv_id"]
        entry = papers.setdefault(
            aid,
            {
                "arxiv_id": aid,
                "title": d.metadata.get("paper_title", "Untitled"),
                "filename": d.metadata.get("source_file", ""),
                "sections": set(),
                "num_chunks": 0,
            },
        )
        entry["sections"].add(d.metadata.get("section", "Unknown"))
        entry["num_chunks"] += 1
    for p in papers.values():
        p["sections"] = sorted(p["sections"])
    return papers
