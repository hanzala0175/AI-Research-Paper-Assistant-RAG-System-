"""
FastAPI backend for the AI Research Paper Assistant.

Endpoints:
  POST   /papers/upload   -> ingest a PDF (section-aware chunking + embed)
  GET    /papers          -> list ingested papers
  DELETE /papers/{arxiv_id} -> remove a paper from the index
  POST   /query           -> citation-grounded Q&A (hybrid retrieval)
  POST   /compare         -> multi-document comparison query
  GET    /health          -> liveness/config check
"""
import shutil
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).resolve().parent.parent))

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware

from app import config
from app.ingestion import ingest_pdf
from app.vectorstore import add_documents, delete_paper, list_papers
from app.rag_chain import answer_question, compare_papers
from app.schemas import (
    UploadResponse,
    PaperInfo,
    QueryRequest,
    QueryResponse,
    CompareRequest,
    CompareResponse,
)

app = FastAPI(
    title="AI Research Paper Assistant (RAG)",
    description="Semantic Q&A over arXiv research papers with hybrid retrieval "
                "and citation-grounded answers, powered by Groq.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
def health():
    return {
        "status": "ok",
        "groq_model": config.GROQ_MODEL,
        "embedding_model": config.EMBEDDING_MODEL,
        "groq_key_configured": bool(config.GROQ_API_KEY),
    }


@app.post("/papers/upload", response_model=UploadResponse)
async def upload_paper(file: UploadFile = File(...)):
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are supported.")

    dest_path = Path(config.PAPERS_DIR) / file.filename
    with open(dest_path, "wb") as f:
        shutil.copyfileobj(file.file, f)

    try:
        result = ingest_pdf(str(dest_path), filename=file.filename)
    except Exception as e:
        dest_path.unlink(missing_ok=True)
        raise HTTPException(status_code=422, detail=f"Failed to process PDF: {e}")

    add_documents(result["documents"])

    return UploadResponse(
        arxiv_id=result["arxiv_id"],
        title=result["title"],
        num_chunks=result["num_chunks"],
        sections=result["sections"],
        message=f"Ingested '{result['title']}' into the index.",
    )


@app.get("/papers", response_model=list[PaperInfo])
def get_papers():
    papers = list_papers()
    return [PaperInfo(**p) for p in papers.values()]


@app.delete("/papers/{arxiv_id}")
def remove_paper(arxiv_id: str):
    deleted = delete_paper(arxiv_id)
    if deleted == 0:
        raise HTTPException(status_code=404, detail="Paper not found in index.")
    return {"deleted_chunks": deleted, "arxiv_id": arxiv_id}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest):
    if not config.GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured on the server.")
    result = answer_question(req.question, arxiv_ids=req.arxiv_ids, top_k=req.top_k)
    return QueryResponse(answer=result["answer"], citations=result["citations"], question=req.question)


@app.post("/compare", response_model=CompareResponse)
def compare(req: CompareRequest):
    if not config.GROQ_API_KEY:
        raise HTTPException(status_code=500, detail="GROQ_API_KEY is not configured on the server.")
    if len(req.arxiv_ids) < 2:
        raise HTTPException(status_code=400, detail="Provide at least 2 arxiv_ids to compare.")
    result = compare_papers(req.question, arxiv_ids=req.arxiv_ids, top_k_per_paper=req.top_k_per_paper or 4)
    return CompareResponse(
        answer=result["answer"],
        citations=result["citations"],
        question=req.question,
        papers_compared=req.arxiv_ids,
    )
