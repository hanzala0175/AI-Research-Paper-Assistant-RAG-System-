# AI Research Paper Assistant (RAG System)

Semantic Q&A over unstructured arXiv research papers, with hybrid retrieval,
section-aware chunking, and citation-grounded answers — built with
**LangChain, FAISS, FastAPI, and Streamlit**, powered by **Groq** for
low-latency LLM inference.

## Features

- **Section-aware chunking** — papers are first segmented by recognized
  headers (Abstract, Introduction, Related Work, Methods, Results,
  Discussion, Conclusion, References, ...) and only then split into
  overlapping chunks *within* each section, so no chunk straddles two
  unrelated parts of the paper. This preserves structure and sharply
  improves retrieval precision vs. naive fixed-size chunking of raw PDF text.
- **Hybrid retrieval** — combines BM25 (sparse/keyword) and FAISS vector
  search (dense/semantic) via LangChain's `EnsembleRetriever`, so exact
  terminology (model names, acronyms, metrics) and paraphrased/semantic
  matches are both captured.
- **Citation-grounded answers** — every answer cites the specific paper
  title, arXiv ID, and section it was pulled from, with inline `[1] [2]`
  markers, to reduce hallucination and make claims verifiable.
- **Multi-document comparison** — ask comparison questions across two or
  more ingested papers; retrieval is balanced per-paper so no single paper
  dominates the context.
- **FastAPI backend + Streamlit frontend** — a REST API for ingestion/query/
  compare, and an interactive UI for uploading papers and chatting with them.
- **Groq-powered generation** — uses Groq's LPU inference for fast, cheap
  generation. Default model: `openai/gpt-oss-120b` (Groq's recommended
  successor to the retired `llama-3.3-70b-versatile`); swap to
  `llama-3.1-8b-instant` for lower latency/cost, or any other chat model
  available on your Groq account.

## Architecture

```
                ┌─────────────────┐
   PDF upload → │  ingestion.py   │  section-aware chunking + metadata
                └────────┬────────┘
                         ▼
                ┌─────────────────┐
                │ vectorstore.py  │  FAISS (persisted) + local
                │                 │  sentence-transformers embeddings
                └────────┬────────┘
                         ▼
                ┌─────────────────┐
   question →   │  retrieval.py   │  EnsembleRetriever(BM25, vector)
                └────────┬────────┘
                         ▼
                ┌─────────────────┐
                │  rag_chain.py   │  citation-grounded prompt → Groq LLM
                └────────┬────────┘
                         ▼
        ┌────────────────┴────────────────┐
        │                                  │
┌───────▼────────┐                ┌────────▼────────┐
│ FastAPI backend │  ◄──REST───►  │ Streamlit frontend│
│   (api/main.py) │                │ (streamlit_app.py)│
└─────────────────┘                └────────────────────┘
```

## Project structure

```
rag-research-assistant/
├── app/
│   ├── config.py        # env-driven configuration
│   ├── ingestion.py      # PDF parsing + section-aware chunking
│   ├── vectorstore.py     # FAISS setup, add/list/delete papers
│   ├── retrieval.py       # hybrid BM25 + vector EnsembleRetriever
│   ├── llm.py              # Groq (ChatGroq) client
│   ├── rag_chain.py         # citation-grounded QA + comparison chains
│   └── schemas.py            # Pydantic request/response models
├── api/
│   └── main.py                # FastAPI app (upload/query/compare/health)
├── streamlit_app.py             # Streamlit UI
├── data/
│   ├── papers/                   # uploaded PDFs land here
│   └── faiss_index/                 # persisted FAISS index (index.faiss + index.pkl)
├── requirements.txt
├── .env.example
└── README.md
```

## Setup

1. **Install dependencies**
   ```bash
   python -m venv .venv && source .venv/bin/activate   # Windows: .venv\Scripts\activate
   pip install -r requirements.txt
   ```

2. **Configure environment**
   ```bash
   cp .env.example .env
   ```
   Edit `.env` and set `GROQ_API_KEY` (free key at
   https://console.groq.com/keys). Adjust `GROQ_MODEL` if you'd like a
   different Groq-hosted model.

3. **Run the backend**
   ```bash
   uvicorn api.main:app --reload --port 8000
   ```
   Interactive API docs: http://localhost:8000/docs

4. **Run the frontend** (in a second terminal)
   ```bash
   streamlit run streamlit_app.py
   ```
   Open the URL Streamlit prints (default http://localhost:8501).

## Using the API directly

**Upload a paper**
```bash
curl -X POST http://localhost:8000/papers/upload \
  -F "file=@/path/to/paper.pdf"
```

**Ask a question**
```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What evaluation datasets were used?"}'
```

**Compare papers**
```bash
curl -X POST http://localhost:8000/compare \
  -H "Content-Type: application/json" \
  -d '{
        "question": "How do the proposed methods differ?",
        "arxiv_ids": ["2301.00001", "2302.00002"]
      }'
```

## Notes & design choices

- **FAISS over ChromaDB**: `faiss-cpu` ships solid prebuilt wheels for
  Windows/Mac/Linux and needs no native build toolchain or server process,
  which avoids the Visual C++ Build Tools / `hnswlib` compilation issues
  ChromaDB can trigger on Windows. The index persists to disk via
  `save_local`/`load_local` and metadata filtering (e.g. restricting a
  query to specific papers) is done with a callable filter rather than
  Chroma's `$in` operator syntax.
- **Embeddings run locally** (`sentence-transformers/all-MiniLM-L6-v2`) since
  Groq does not currently serve an embeddings endpoint. This also keeps
  ingestion free and fast; only the generation step calls the Groq API.
- **BM25 index is rebuilt per query** from the current FAISS docstore contents
  (optionally filtered to selected papers). This is inexpensive for typical
  corpora (hundreds–thousands of chunks) and guarantees the sparse index
  never drifts out of sync with additions/deletions — no separate reindex
  step required.
- **Model deprecations**: Groq periodically retires models (e.g.
  `llama-3.3-70b-versatile` was deprecated in favor of `openai/gpt-oss-120b`
  and `qwen/qwen3.6-27b`). If you hit a `model_decommissioned` error, check
  https://console.groq.com/docs/models and update `GROQ_MODEL` in `.env`.

## Possible extensions

- Swap FAISS for a managed vector DB (Pinecone/Weaviate/Chroma server mode) for multi-user
  deployments.
- Add streaming responses (Groq supports streaming; wire `ChatGroq(streaming=True)`
  through to a Streamlit `st.write_stream`).
- Add re-ranking (e.g. a cross-encoder) after hybrid retrieval for even
  higher precision on large corpora.
- Persist chat history per session for follow-up questions.
