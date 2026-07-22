"""
Citation-grounded RAG chains:
  - answer_question(): single/multi-paper Q&A where every claim in the
    answer is tied back to (paper title, section, arXiv ID) to reduce
    hallucination.
  - compare_papers(): retrieves top-k chunks *per paper* (so smaller/less
    similar papers aren't drowned out by a single dominant paper) and asks
    the LLM to produce a structured, citation-grounded comparison.
"""
from typing import List, Dict, Optional
from collections import defaultdict

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

from app.llm import get_llm
from app.retrieval import get_hybrid_retriever, retrieve
from app.schemas import Citation


QA_SYSTEM_PROMPT = """You are a meticulous AI research assistant that answers questions about \
academic papers using ONLY the provided context excerpts.

Rules:
1. Base your answer strictly on the CONTEXT below. Do not use outside knowledge.
2. If the context does not contain enough information to answer, say so plainly \
instead of guessing.
3. After every factual claim, include an inline citation marker like [1], [2] \
that refers to the numbered source excerpt(s) below the context supports it with.
4. Be precise and technical where the source material is technical; do not oversimplify \
equations, model names, or numbers.
5. If multiple papers are present in the context and they disagree, point that out explicitly.

CONTEXT EXCERPTS:
{context_block}

Answer the user's question using the rules above."""

COMPARE_SYSTEM_PROMPT = """You are an AI research assistant performing a multi-paper comparison.

You are given excerpts from {num_papers} different papers. Compare them with respect to the \
user's question. Structure your answer with:
  - A short summary of each paper's relevant position/approach (one paragraph each)
  - A "Key Similarities" section
  - A "Key Differences" section
Cite every claim with an inline marker like [1], [2] referring to the numbered excerpts below, \
and do not introduce information that isn't in the excerpts.

CONTEXT EXCERPTS (grouped by paper):
{context_block}

Now answer the user's comparison question using the rules above."""


def _format_context_block(docs: List[Document]) -> str:
    lines = []
    for i, d in enumerate(docs, start=1):
        m = d.metadata
        lines.append(
            f"[{i}] Paper: \"{m.get('paper_title', 'Unknown')}\" "
            f"(arXiv:{m.get('arxiv_id', 'unknown')}) — Section: {m.get('section', 'Unknown')}\n"
            f"{d.page_content.strip()}\n"
        )
    return "\n".join(lines)


def _docs_to_citations(docs: List[Document]) -> List[Citation]:
    citations = []
    for d in docs:
        m = d.metadata
        snippet = d.page_content.strip().replace("\n", " ")
        if len(snippet) > 220:
            snippet = snippet[:220].rsplit(" ", 1)[0] + "..."
        citations.append(
            Citation(
                paper_title=m.get("paper_title", "Unknown"),
                arxiv_id=m.get("arxiv_id", "unknown"),
                section=m.get("section", "Unknown"),
                chunk_id=m.get("chunk_id", ""),
                snippet=snippet,
            )
        )
    return citations


def answer_question(
    question: str,
    arxiv_ids: Optional[List[str]] = None,
    top_k: Optional[int] = None,
) -> Dict:
    """Standard citation-grounded QA over one or more papers (hybrid retrieval)."""
    docs = retrieve(question, arxiv_ids=arxiv_ids, top_k=top_k)
    if not docs:
        return {
            "answer": "I couldn't find any relevant content in the ingested papers to answer this question.",
            "citations": [],
        }

    context_block = _format_context_block(docs)
    prompt = ChatPromptTemplate.from_messages(
        [("system", QA_SYSTEM_PROMPT), ("human", "{question}")]
    )
    chain = prompt | get_llm() | StrOutputParser()
    answer = chain.invoke({"context_block": context_block, "question": question})

    return {"answer": answer, "citations": _docs_to_citations(docs)}


def compare_papers(
    question: str,
    arxiv_ids: List[str],
    top_k_per_paper: int = 4,
) -> Dict:
    """
    Multi-document comparison: retrieves top_k_per_paper chunks *per paper*
    (rather than one global top-k) so every paper gets fair representation
    in the comparison, then asks the LLM to compare them structurally.
    """
    if len(arxiv_ids) < 2:
        raise ValueError("compare_papers requires at least 2 arxiv_ids")

    per_paper_docs: Dict[str, List[Document]] = {}
    all_docs: List[Document] = []
    for aid in arxiv_ids:
        retriever = get_hybrid_retriever(arxiv_ids=[aid], top_k=top_k_per_paper)
        docs = retriever.invoke(question)
        per_paper_docs[aid] = docs
        all_docs.extend(docs)

    if not all_docs:
        return {
            "answer": "I couldn't find relevant content for the requested papers.",
            "citations": [],
        }

    context_block = _format_context_block(all_docs)
    prompt = ChatPromptTemplate.from_messages(
        [("system", COMPARE_SYSTEM_PROMPT), ("human", "{question}")]
    )
    chain = prompt | get_llm() | StrOutputParser()
    answer = chain.invoke(
        {
            "context_block": context_block,
            "question": question,
            "num_papers": len(arxiv_ids),
        }
    )

    return {"answer": answer, "citations": _docs_to_citations(all_docs)}
