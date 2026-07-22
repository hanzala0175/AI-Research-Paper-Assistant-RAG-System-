"""
Streamlit frontend for the AI Research Paper Assistant.

Talks to the FastAPI backend (run separately: `uvicorn api.main:app`).
Set API_BASE_URL below (or via env var) if the backend runs elsewhere.
"""
import os
import requests
import streamlit as st

API_BASE_URL = os.getenv("API_BASE_URL", "http://localhost:8000")

st.set_page_config(page_title="AI Research Paper Assistant", page_icon="📄", layout="wide")


# ---------- Helpers ----------
def api_get(path):
    r = requests.get(f"{API_BASE_URL}{path}", timeout=30)
    r.raise_for_status()
    return r.json()


def api_post(path, json=None, files=None):
    r = requests.post(f"{API_BASE_URL}{path}", json=json, files=files, timeout=120)
    if not r.ok:
        raise RuntimeError(r.json().get("detail", r.text))
    return r.json()


def api_delete(path):
    r = requests.delete(f"{API_BASE_URL}{path}", timeout=30)
    r.raise_for_status()
    return r.json()


def render_citations(citations):
    if not citations:
        return
    with st.expander(f"📚 Sources ({len(citations)})", expanded=False):
        for i, c in enumerate(citations, start=1):
            st.markdown(
                f"**[{i}] {c['paper_title']}** — arXiv:`{c['arxiv_id']}` · Section: *{c['section']}*"
            )
            st.caption(c["snippet"])
            st.divider()


# ---------- Sidebar: corpus management ----------
st.sidebar.title("📄 Research Paper Assistant")
st.sidebar.caption("RAG over arXiv papers · LangChain + ChromaDB + Groq")

try:
    health = api_get("/health")
    st.sidebar.success(f"Backend up · model: `{health['groq_model']}`")
    if not health["groq_key_configured"]:
        st.sidebar.error("GROQ_API_KEY not set on the backend — Q&A will fail until it is.")
except Exception:
    st.sidebar.error(f"Cannot reach backend at {API_BASE_URL}. Start it with:\n\n"
                      "`uvicorn api.main:app --reload`")
    st.stop()

st.sidebar.subheader("Upload a paper")
uploaded_file = st.sidebar.file_uploader("Upload arXiv PDF", type=["pdf"])
if uploaded_file is not None:
    if st.sidebar.button("Ingest paper", use_container_width=True):
        with st.sidebar.status("Processing PDF (section-aware chunking + embedding)..."):
            try:
                resp = api_post(
                    "/papers/upload",
                    files={"file": (uploaded_file.name, uploaded_file.getvalue(), "application/pdf")},
                )
                st.sidebar.success(f"Ingested: {resp['title']} ({resp['num_chunks']} chunks)")
            except Exception as e:
                st.sidebar.error(f"Failed: {e}")

st.sidebar.subheader("Ingested papers")
try:
    papers = api_get("/papers")
except Exception:
    papers = []

if not papers:
    st.sidebar.info("No papers ingested yet. Upload a PDF above to get started.")
else:
    for p in papers:
        with st.sidebar.container(border=True):
            st.markdown(f"**{p['title']}**")
            st.caption(f"arXiv:{p['arxiv_id']} · {p['num_chunks']} chunks")
            if st.button("🗑️ Remove", key=f"del-{p['arxiv_id']}"):
                api_delete(f"/papers/{p['arxiv_id']}")
                st.rerun()

# ---------- Main area ----------
tab_qa, tab_compare = st.tabs(["🔎 Ask a question", "⚖️ Compare papers"])

with tab_qa:
    st.subheader("Semantic Q&A over your papers")
    if not papers:
        st.info("Upload at least one paper in the sidebar to start asking questions.")
    else:
        paper_options = {f"{p['title']} ({p['arxiv_id']})": p["arxiv_id"] for p in papers}
        scope = st.multiselect(
            "Restrict to specific papers (leave empty to search the whole corpus)",
            options=list(paper_options.keys()),
        )
        question = st.text_area("Your question", placeholder="e.g. What datasets did the authors use for evaluation?")
        if st.button("Ask", type="primary", disabled=not question.strip()):
            arxiv_ids = [paper_options[s] for s in scope] if scope else None
            with st.spinner("Retrieving relevant chunks and generating a grounded answer..."):
                try:
                    result = api_post("/query", json={"question": question, "arxiv_ids": arxiv_ids})
                    st.markdown(result["answer"])
                    render_citations(result["citations"])
                except Exception as e:
                    st.error(f"Query failed: {e}")

with tab_compare:
    st.subheader("Multi-document comparison")
    if len(papers) < 2:
        st.info("Upload at least 2 papers to use comparison mode.")
    else:
        paper_options = {f"{p['title']} ({p['arxiv_id']})": p["arxiv_id"] for p in papers}
        chosen = st.multiselect("Papers to compare (select 2+)", options=list(paper_options.keys()))
        compare_q = st.text_area(
            "Comparison question",
            placeholder="e.g. How do these papers' proposed methods differ, and which reports stronger results?",
        )
        if st.button("Compare", type="primary", disabled=len(chosen) < 2 or not compare_q.strip()):
            arxiv_ids = [paper_options[c] for c in chosen]
            with st.spinner("Retrieving evidence from each paper and generating a comparison..."):
                try:
                    result = api_post("/compare", json={"question": compare_q, "arxiv_ids": arxiv_ids})
                    st.markdown(result["answer"])
                    render_citations(result["citations"])
                except Exception as e:
                    st.error(f"Comparison failed: {e}")
