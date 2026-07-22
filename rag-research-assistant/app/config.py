"""
Central configuration for the RAG Research Paper Assistant.
All values are overridable via environment variables / .env file.
"""
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

BASE_DIR = Path(__file__).resolve().parent.parent

# --- Groq LLM ---
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
# openai/gpt-oss-120b is Groq's recommended migration target for the
# retired llama-3.3-70b-versatile model and gives the strongest reasoning
# quality for citation-grounded QA. Swap to llama-3.1-8b-instant for speed.
GROQ_MODEL = os.getenv("GROQ_MODEL", "openai/gpt-oss-120b")
GROQ_TEMPERATURE = float(os.getenv("GROQ_TEMPERATURE", "0.1"))

# --- Embeddings (local — Groq does not serve an embeddings endpoint) ---
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")

# --- Storage ---
FAISS_INDEX_DIR = str(BASE_DIR / os.getenv("FAISS_INDEX_DIR", "./data/faiss_index"))
PAPERS_DIR = str(BASE_DIR / os.getenv("PAPERS_DIR", "./data/papers"))

# --- Chunking ---
CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

# --- Retrieval ---
TOP_K = int(os.getenv("TOP_K", "6"))
BM25_WEIGHT = float(os.getenv("BM25_WEIGHT", "0.4"))
VECTOR_WEIGHT = float(os.getenv("VECTOR_WEIGHT", "0.6"))

# Ensure directories exist
Path(FAISS_INDEX_DIR).mkdir(parents=True, exist_ok=True)
Path(PAPERS_DIR).mkdir(parents=True, exist_ok=True)
