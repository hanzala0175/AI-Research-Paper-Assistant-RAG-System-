"""
Ingestion pipeline for arXiv research papers.

Design goals (per project spec):
  - Section-aware chunking: split on recognized paper sections (Abstract,
    Introduction, Related Work, Methods, Results, Discussion, Conclusion,
    References, ...) BEFORE running the recursive text splitter, so a chunk
    never straddles two unrelated sections. This preserves document
    structure and gives the retriever much more precise, citeable hits than
    naive fixed-size chunking of the raw PDF text.
  - Every chunk is tagged with metadata (arxiv_id, paper title, section,
    page range, chunk_id) so downstream answers can be citation-grounded.
"""
import re
import uuid
from pathlib import Path
from typing import List, Dict, Optional

from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

from app import config

# Canonical section headers commonly found in arXiv / academic papers.
# Matched case-insensitively at the start of a line, optionally numbered
# ("1. Introduction", "II. Related Work", "3 Methodology").
SECTION_PATTERNS = [
    "abstract",
    "introduction",
    "background",
    "related work",
    "motivation",
    "problem statement",
    "method", "methods", "methodology", "approach",
    "model", "architecture",
    "experiment", "experiments", "experimental setup", "experimental results",
    "results",
    "evaluation",
    "discussion",
    "analysis",
    "ablation", "ablation study",
    "limitations",
    "future work",
    "conclusion", "conclusions",
    "acknowledgments", "acknowledgements",
    "references", "bibliography",
    "appendix",
]

_HEADER_REGEX = re.compile(
    r"^\s*(?:[IVXLCM]+\.|\d+(?:\.\d+)*\.?)?\s*("
    + "|".join(sorted(SECTION_PATTERNS, key=len, reverse=True))
    + r")\s*$",
    re.IGNORECASE,
)

_ARXIV_ID_REGEX = re.compile(r"\b(\d{4}\.\d{4,5}(?:v\d+)?)\b")


def extract_arxiv_id(text: str, filename: str) -> str:
    """Look for an arXiv identifier in the first page text, else the filename,
    else fall back to a generated stub id."""
    match = _ARXIV_ID_REGEX.search(text[:2000])
    if match:
        return match.group(1)
    match = _ARXIV_ID_REGEX.search(filename)
    if match:
        return match.group(1)
    return f"local-{uuid.uuid4().hex[:8]}"


def extract_title(reader: PdfReader, first_page_text: str) -> str:
    """Prefer PDF metadata title; fall back to the first non-trivial line
    of the first page (typical arXiv paper layout: title is the first
    prominent line)."""
    meta_title = (reader.metadata.title if reader.metadata else None) or ""
    meta_title = meta_title.strip()
    if meta_title and len(meta_title) > 5 and "untitled" not in meta_title.lower():
        return meta_title

    for line in first_page_text.splitlines():
        line = line.strip()
        if len(line) > 8 and not _ARXIV_ID_REGEX.search(line):
            return line
    return "Untitled Paper"


def _split_into_sections(full_text: str) -> List[Dict[str, str]]:
    """Walk the extracted text line by line, opening a new 'section' whenever
    a recognized header is encountered. Returns an ordered list of
    {section_name, text} blocks."""
    sections: List[Dict[str, str]] = []
    current_name = "Front Matter"
    current_lines: List[str] = []

    for raw_line in full_text.splitlines():
        header_match = _HEADER_REGEX.match(raw_line.strip())
        if header_match:
            # flush current section
            if current_lines:
                sections.append({"section": current_name, "text": "\n".join(current_lines).strip()})
            current_name = header_match.group(1).title()
            current_lines = []
        else:
            current_lines.append(raw_line)

    if current_lines:
        sections.append({"section": current_name, "text": "\n".join(current_lines).strip()})

    # Merge out empty/near-empty sections into neighbors to avoid junk chunks
    merged: List[Dict[str, str]] = []
    for sec in sections:
        if len(sec["text"]) < 40 and merged:
            merged[-1]["text"] += "\n" + sec["text"]
        else:
            merged.append(sec)
    return merged


def load_pdf_text_by_page(pdf_path: str) -> List[str]:
    reader = PdfReader(pdf_path)
    return [page.extract_text() or "" for page in reader.pages], reader


def section_aware_chunk_pdf(
    pdf_path: str,
    filename: Optional[str] = None,
    chunk_size: int = None,
    chunk_overlap: int = None,
) -> List[Document]:
    """
    Parse a PDF, segment it by recognized section headers, then apply a
    recursive character splitter *within* each section so chunks never
    cross a section boundary. Returns a list of LangChain Documents with
    rich metadata for citation grounding.
    """
    chunk_size = chunk_size or config.CHUNK_SIZE
    chunk_overlap = chunk_overlap or config.CHUNK_OVERLAP
    filename = filename or Path(pdf_path).name

    pages, reader = load_pdf_text_by_page(pdf_path)
    full_text = "\n".join(pages)

    arxiv_id = extract_arxiv_id(full_text, filename)
    title = extract_title(reader, pages[0] if pages else "")

    sections = _split_into_sections(full_text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    documents: List[Document] = []
    for sec_idx, sec in enumerate(sections):
        if not sec["text"].strip():
            continue
        sub_chunks = splitter.split_text(sec["text"])
        for chunk_idx, chunk_text in enumerate(sub_chunks):
            chunk_id = f"{arxiv_id}::{sec_idx}-{chunk_idx}"
            documents.append(
                Document(
                    page_content=chunk_text,
                    metadata={
                        "arxiv_id": arxiv_id,
                        "paper_title": title,
                        "section": sec["section"],
                        "chunk_id": chunk_id,
                        "source_file": filename,
                    },
                )
            )
    return documents


def ingest_pdf(pdf_path: str, filename: Optional[str] = None) -> Dict:
    """High-level convenience wrapper used by the API layer."""
    docs = section_aware_chunk_pdf(pdf_path, filename=filename)
    if not docs:
        raise ValueError("No extractable text found in PDF.")
    sections = sorted({d.metadata["section"] for d in docs})
    return {
        "documents": docs,
        "arxiv_id": docs[0].metadata["arxiv_id"],
        "title": docs[0].metadata["paper_title"],
        "sections": sections,
        "num_chunks": len(docs),
    }
