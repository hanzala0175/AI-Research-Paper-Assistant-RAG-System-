"""Pydantic schemas shared across the FastAPI backend."""
from typing import List, Optional
from pydantic import BaseModel, Field


class PaperInfo(BaseModel):
    arxiv_id: str
    title: str
    filename: str
    num_chunks: int
    sections: List[str]


class UploadResponse(BaseModel):
    arxiv_id: str
    title: str
    num_chunks: int
    sections: List[str]
    message: str


class Citation(BaseModel):
    paper_title: str
    arxiv_id: str
    section: str
    chunk_id: str
    snippet: str


class QueryRequest(BaseModel):
    question: str = Field(..., description="Natural language question about the ingested papers")
    arxiv_ids: Optional[List[str]] = Field(
        default=None,
        description="Optional: restrict retrieval to specific papers (by arXiv ID). "
                    "Omit to search across the whole corpus.",
    )
    top_k: Optional[int] = Field(default=None, description="Override number of chunks retrieved")


class QueryResponse(BaseModel):
    answer: str
    citations: List[Citation]
    question: str


class CompareRequest(BaseModel):
    question: str = Field(..., description="What to compare across the papers, e.g. 'How do these papers' methods differ?'")
    arxiv_ids: List[str] = Field(..., min_length=2, description="Two or more arXiv IDs to compare")
    top_k_per_paper: Optional[int] = Field(default=4, description="Chunks retrieved per paper")


class CompareResponse(BaseModel):
    answer: str
    citations: List[Citation]
    question: str
    papers_compared: List[str]
