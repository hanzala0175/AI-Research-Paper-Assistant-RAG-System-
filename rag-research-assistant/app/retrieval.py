"""
Hybrid retrieval: BM25 (sparse, keyword-based) + Chroma vector search
(dense, semantic), combined with LangChain's EnsembleRetriever.

Why hybrid: vector search alone can miss exact terminology (model names,
acronyms, equation symbols, numeric hyperparameters) that BM25 nails, while
BM25 alone misses paraphrased/semantically-related passages that vector
search excels at. Combining both consistently improves recall for
research-paper QA.
"""
from typing import List, Optional

from langchain_community.retrievers import BM25Retriever
from langchain.retrievers import EnsembleRetriever
from langchain_core.documents import Document
from langchain_core.retrievers import BaseRetriever

from app import config
from app.vectorstore import get_vectorstore, get_all_documents, metadata_filter


def _build_bm25_retriever(documents: List[Document], k: int) -> Optional[BM25Retriever]:
    if not documents:
        return None
    bm25 = BM25Retriever.from_documents(documents)
    bm25.k = k
    return bm25


def get_hybrid_retriever(
    arxiv_ids: Optional[List[str]] = None,
    top_k: Optional[int] = None,
) -> BaseRetriever:
    """
    Build an EnsembleRetriever combining:
      - BM25Retriever over the (optionally paper-filtered) corpus
      - Chroma similarity search (optionally filtered by arxiv_id via `where`)

    Rebuilding BM25 per-query is cheap for typical corpora (hundreds to a
    few thousand chunks) and guarantees it always reflects the latest
    ingested/deleted papers without a separate sync step.
    """
    k = top_k or config.TOP_K

    corpus = get_all_documents(arxiv_ids=arxiv_ids)
    bm25_retriever = _build_bm25_retriever(corpus, k=k)

    vs = get_vectorstore()
    search_kwargs = {"k": k}
    filter_fn = metadata_filter(arxiv_ids)
    if filter_fn:
        search_kwargs["filter"] = filter_fn
    vector_retriever = vs.as_retriever(search_kwargs=search_kwargs)

    if bm25_retriever is None:
        # No documents yet / filter matched nothing sparse-side: fall back
        # to vector-only so the API doesn't error on an empty corpus.
        return vector_retriever

    return EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[config.BM25_WEIGHT, config.VECTOR_WEIGHT],
    )


def retrieve(question: str, arxiv_ids: Optional[List[str]] = None, top_k: Optional[int] = None) -> List[Document]:
    retriever = get_hybrid_retriever(arxiv_ids=arxiv_ids, top_k=top_k)
    return retriever.invoke(question)
