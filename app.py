"""
Paracetamol Document QA System - RAG Application (OLLAMA VERSION)
100% LOCAL — No API keys. Auto-detects installed models for embeddings.
Works with only llama3 installed — no nomic-embed-text required.
"""

import streamlit as st
import os
import tempfile
import logging
import requests
import json
from typing import List

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.language_models.llms import LLM

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# ============================================================================
# CONFIG
# ============================================================================

OLLAMA_BASE_URL = os.getenv("OLLAMA_HOST", "http://localhost:11434")

CONFIG = {
    "chunk_size": 1000,
    "chunk_overlap": 200,
    "llm_model": "llama3:latest",
    "embedding_model": "llama3:latest",
    "embedding_mode": "ollama",
    "temperature": 0.5,
    "max_tokens": 1024,
    "retrieval_k": 3,
    "ollama_url": OLLAMA_BASE_URL,
}

KNOWN_EMBED_MODELS = [
    "nomic-embed-text",
    "mxbai-embed-large",
    "all-minilm",
    "snowflake-arctic-embed",
]

ALL_CHAT_MODELS = [
    "llama3.2","llama3.1","llama3","llama2","mistral","mistral-nemo",
    "gemma3","gemma2","gemma","phi4","phi3","qwen2.5","qwen2",
    "deepseek-r1","deepseek-r1:8b","codellama","neural-chat",
    "openchat","vicuna","orca-mini",
]

# ============================================================================
# OLLAMA HELPERS
# ============================================================================

def check_ollama(url):
    try:
        return requests.get(f"{url}/api/tags", timeout=3).status_code == 200
    except Exception:
        return False


def get_installed(url):
    try:
        r = requests.get(f"{url}/api/tags", timeout=5)
        if r.status_code == 200:
            return [m["name"] for m in r.json().get("models", [])]
    except Exception:
        pass
    return []


def pull_stream(model, url):
    with requests.post(f"{url}/api/pull", json={"name": model},
                       stream=True, timeout=600) as r:
        for line in r.iter_lines():
            if line:
                try: yield json.loads(line)
                except Exception: pass


def best_embed(installed):
    for m in installed:
        if any(em in m for em in KNOWN_EMBED_MODELS):
            return m, "ollama"
    if installed:
        return installed[0], "ollama"
    return "sentence-transformers/all-MiniLM-L6-v2", "huggingface"


# ============================================================================
# CUSTOM EMBEDDINGS
# ============================================================================

class OllamaEmbeddings(Embeddings):
    def __init__(self, model: str, base_url: str = "http://localhost:11434"):
        self.model    = model
        self.base_url = base_url

    def _embed_one(self, text: str) -> List[float]:
        r = requests.post(
            f"{self.base_url}/api/embeddings",
            json={"model": self.model, "prompt": text},
            timeout=120,
        )
        if r.status_code == 404:
            raise Exception(f"Model `{self.model}` not found.\nPull it with:  ollama pull {self.model}")
        r.raise_for_status()
        return r.json()["embedding"]

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return [self._embed_one(t) for t in texts]

    def embed_query(self, text: str) -> List[float]:
        return self._embed_one(text)


# ============================================================================
# OLLAMA LLM
# ============================================================================

class OllamaLLM(LLM):
    model_name: str = "llama3:latest"
    temperature: float = 0.5
    max_tokens: int = 1024
    base_url: str = "http://localhost:11434"

    class Config:
        arbitrary_types_allowed = True
        extra = "allow"

    def __init__(self, model_name="llama3:latest", temperature=0.5,
                 max_tokens=1024, base_url="http://localhost:11434", **kwargs):
        super().__init__(model_name=model_name, temperature=temperature,
                         max_tokens=max_tokens, base_url=base_url, **kwargs)

    @property
    def _llm_type(self): return "ollama"

    def _call(self, prompt: str, stop=None, **kwargs) -> str:
        try:
            r = requests.post(
                f"{self.base_url}/api/generate",
                json={
                    "model": self.model_name,
                    "prompt": prompt,
                    "stream": False,
                    "options": {
                        "temperature": self.temperature,
                        "num_predict": self.max_tokens,
                    },
                },
                timeout=300,
            )
            if r.status_code == 404:
                raise Exception(f"Model `{self.model_name}` not found.\nRun:  ollama pull {self.model_name}")
            r.raise_for_status()
            return r.json().get("response", "").strip()
        except requests.exceptions.ConnectionError:
            raise Exception(f"Cannot connect to Ollama at {self.base_url}.\nStart it with:  ollama serve")


# ============================================================================
# DOCUMENT PROCESSING
# ============================================================================

def process_document(uploaded_file) -> List[Document]:
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.getvalue())
        tmp_path = tmp.name
    try:
        docs = PyPDFLoader(tmp_path).load()
        if not docs:
            raise ValueError("PDF has no readable pages.")
        return RecursiveCharacterTextSplitter(
            chunk_size=CONFIG["chunk_size"],
            chunk_overlap=CONFIG["chunk_overlap"],
            separators=["\n\n", "\n", " ", ""],
        ).split_documents(docs)
    except Exception as e:
        raise Exception(f"PDF processing failed: {str(e)}")
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


def make_embeddings():
    mode  = CONFIG["embedding_mode"]
    model = CONFIG["embedding_model"]
    if mode == "huggingface":
        from langchain_community.embeddings import HuggingFaceEmbeddings
        return HuggingFaceEmbeddings(
            model_name=model,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
    return OllamaEmbeddings(model=model, base_url=CONFIG["ollama_url"])


def create_vector_db(chunks: List[Document]) -> FAISS:
    try:
        return FAISS.from_documents(chunks, make_embeddings())
    except Exception as e:
        raise Exception(f"Embedding error with `{CONFIG['embedding_model']}`: {str(e)}")


def fmt(docs): return "\n\n".join(d.page_content for d in docs)


# ============================================================================
# STREAMLIT PAGE CONFIG — REDESIGNED FRONTEND
# ============================================================================

st.set_page_config(
    page_title="Paracetamol RAG — Ollama",
    page_icon="🦙",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@300;400;500;600;700&family=JetBrains+Mono:wght@300;400;500&display=swap');

*, *::before, *::after { box-sizing: border-box; }

html, body, [data-testid="stAppViewContainer"] {
    background: #0f0f0f !important;
    font-family: 'JetBrains Mono', monospace !important;
    color: #d4d4d4 !important;
}

#MainMenu, footer, header { visibility: hidden; }
[data-testid="stToolbar"] { display: none; }

/* ── Sidebar ── */
[data-testid="stSidebar"] {
    background: #141414 !important;
    border-right: 1px solid #222 !important;
}
[data-testid="stSidebar"] * {
    color: #888 !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {
    color: #a3e635 !important;
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
}
[data-testid="stSidebar"] .stTextInput input {
    background: #0a0a0a !important;
    border: 1px solid #2a2a2a !important;
    color: #d4d4d4 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    border-radius: 3px !important;
}
[data-testid="stSidebar"] .stTextInput input:focus {
    border-color: #a3e635 !important;
    outline: none !important;
    box-shadow: 0 0 0 2px rgba(163,230,53,0.07) !important;
}
[data-testid="stSidebar"] .stSelectbox > div > div {
    background: #0a0a0a !important;
    border: 1px solid #2a2a2a !important;
    color: #d4d4d4 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    border-radius: 3px !important;
}
[data-testid="stSidebar"] .stSlider { padding: 0.2rem 0; }
[data-testid="stSidebar"] .stSlider [data-testid="stThumbValue"] {
    color: #a3e635 !important;
    font-size: 0.68rem !important;
}

/* ── Main container ── */
.main .block-container {
    padding: 2rem 2.5rem !important;
    max-width: 1080px !important;
}

/* ── Hero ── */
.hero {
    padding: 2.5rem 0 2rem;
    border-bottom: 1px solid #1e1e1e;
    margin-bottom: 2rem;
}
.hero-eyebrow {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.62rem;
    color: #a3e635;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    margin-bottom: 0.6rem;
}
.hero-title {
    font-family: 'Space Grotesk', sans-serif !important;
    font-size: 2.8rem !important;
    font-weight: 700 !important;
    color: #f0f0f0 !important;
    line-height: 1.05 !important;
    letter-spacing: -0.03em !important;
    margin: 0 !important;
}
.hero-title em {
    font-style: normal;
    color: #a3e635;
}
.hero-tagline {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.74rem;
    color: #444;
    margin-top: 0.6rem;
    letter-spacing: 0.04em;
}
.hero-tagline span { color: #666; }

/* ── Status dot ── */
.status-row { display: flex; align-items: center; gap: 0.5rem; margin: 1rem 0; }
.dot {
    width: 8px; height: 8px;
    border-radius: 50%;
    display: inline-block;
    flex-shrink: 0;
}
.dot-green { background: #a3e635; box-shadow: 0 0 6px #a3e635aa; }
.dot-red   { background: #f43f5e; box-shadow: 0 0 6px #f43f5eaa; }
.dot-amber { background: #f59e0b; box-shadow: 0 0 6px #f59e0baa; }
.status-text {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #666;
    letter-spacing: 0.04em;
}

/* ── Section label ── */
.sec {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.22em;
    text-transform: uppercase;
    color: #333;
    margin: 2rem 0 0.8rem;
    display: flex;
    align-items: center;
    gap: 0.75rem;
}
.sec::after { content:''; flex:1; height:1px; background:#1e1e1e; }

/* ── File uploader ── */
[data-testid="stFileUploader"] {
    background: #111 !important;
    border: 1.5px dashed #2a2a2a !important;
    border-radius: 6px !important;
    padding: 1.25rem !important;
    transition: border-color 0.2s;
}
[data-testid="stFileUploader"]:hover { border-color: #a3e635 !important; }
[data-testid="stFileUploader"] label {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #a3e635 !important;
    font-size: 0.82rem !important;
    font-weight: 600 !important;
}
[data-testid="stFileUploaderDropzoneInstructions"] { color: #444 !important; font-size: 0.72rem !important; }

/* ── Text input ── */
[data-testid="stTextInput"] input {
    background: #111 !important;
    border: 1px solid #2a2a2a !important;
    border-radius: 4px !important;
    color: #d4d4d4 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.8rem !important;
    padding: 0.7rem 1rem !important;
    transition: border-color 0.2s, box-shadow 0.2s;
}
[data-testid="stTextInput"] input:focus {
    border-color: #a3e635 !important;
    box-shadow: 0 0 0 3px rgba(163,230,53,0.06) !important;
    outline: none !important;
}
[data-testid="stTextInput"] label {
    font-family: 'Space Grotesk', sans-serif !important;
    color: #a3e635 !important;
    font-size: 0.68rem !important;
    font-weight: 600 !important;
    letter-spacing: 0.16em !important;
    text-transform: uppercase !important;
}

/* ── Answer card ── */
.answer-wrap {
    background: #111;
    border: 1px solid #1e1e1e;
    border-left: 3px solid #a3e635;
    border-radius: 6px;
    padding: 1.5rem 1.75rem;
    margin: 1.25rem 0;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.82rem;
    color: #c8c8c8;
    line-height: 1.8;
}
.answer-tag {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.6rem;
    font-weight: 600;
    letter-spacing: 0.2em;
    text-transform: uppercase;
    color: #a3e635;
    margin-bottom: 0.75rem;
    display: flex;
    align-items: center;
    gap: 0.6rem;
}
.answer-tag::after { content:''; flex:1; height:1px; background:#1e1e1e; }

/* ── Model badge ── */
.model-badge {
    display: inline-flex;
    align-items: center;
    gap: 0.4rem;
    background: #161616;
    border: 1px solid #2a2a2a;
    border-radius: 3px;
    padding: 0.3rem 0.75rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #a3e635;
    letter-spacing: 0.06em;
    margin-bottom: 1rem;
}

/* ── Expanders ── */
[data-testid="stExpander"] {
    background: #0c0c0c !important;
    border: 1px solid #1a1a1a !important;
    border-radius: 4px !important;
    margin-bottom: 0.35rem !important;
}
[data-testid="stExpander"] summary {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    color: #444 !important;
    letter-spacing: 0.06em !important;
    padding: 0.55rem 1rem !important;
}
[data-testid="stExpander"] summary:hover { color: #a3e635 !important; }
[data-testid="stExpander"] .streamlit-expanderContent {
    background: #080808 !important;
    font-size: 0.73rem !important;
    color: #666 !important;
    font-family: 'JetBrains Mono', monospace !important;
    line-height: 1.75 !important;
    padding: 1rem 1.25rem !important;
}

/* ── Streamlit status/alert boxes ── */
[data-testid="stSuccess"] {
    background: #0a120a !important;
    border: 1px solid #1a3a1a !important;
    color: #4a9a4a !important;
    font-size: 0.75rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 4px !important;
}
[data-testid="stError"] {
    background: #140808 !important;
    border: 1px solid #3a1010 !important;
    color: #c04040 !important;
    font-size: 0.75rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 4px !important;
}
[data-testid="stWarning"] {
    background: #120e02 !important;
    border: 1px solid #3a2a00 !important;
    color: #b08000 !important;
    font-size: 0.75rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 4px !important;
}
[data-testid="stInfo"] {
    background: #080c14 !important;
    border: 1px solid #101a2a !important;
    color: #4070a0 !important;
    font-size: 0.75rem !important;
    font-family: 'JetBrains Mono', monospace !important;
    border-radius: 4px !important;
}

/* ── Status expander in sidebar ── */
[data-testid="stSidebar"] [data-testid="stExpander"] {
    background: #0c0c0c !important;
    border: 1px solid #1e1e1e !important;
}
[data-testid="stSidebar"] [data-testid="stExpander"] summary {
    color: #555 !important;
    font-size: 0.68rem !important;
}

/* ── Buttons ── */
[data-testid="stButton"] button {
    background: #0a0a0a !important;
    border: 1px solid #2a2a2a !important;
    color: #888 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.68rem !important;
    border-radius: 3px !important;
    transition: all 0.15s;
}
[data-testid="stButton"] button:hover {
    border-color: #a3e635 !important;
    color: #a3e635 !important;
}

/* ── Spinner ── */
[data-testid="stSpinner"] { color: #a3e635 !important; }

/* ── Config kv grid ── */
.kv { display:flex; justify-content:space-between; align-items:center;
      background:#0c0c0c; border:1px solid #161616; border-radius:3px;
      padding:0.4rem 0.75rem; margin-bottom:0.3rem;
      font-family:'JetBrains Mono',monospace; font-size:0.67rem; }
.kv-k { color:#333; }
.kv-v { color:#a3e635; }

/* ── Pipeline grid ── */
.pipe {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(130px, 1fr));
    gap: 0.6rem;
    margin: 0.75rem 0;
}
.pipe-node {
    background: #0c0c0c;
    border: 1px solid #1a1a1a;
    border-radius: 4px;
    padding: 0.7rem 0.9rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.65rem;
    color: #444;
}
.pipe-node strong {
    display:block;
    color: #a3e635;
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    margin-bottom: 0.2rem;
}

/* ── Onboarding ── */
.ob-grid {
    display: grid;
    grid-template-columns: 1fr 1fr;
    gap: 0.75rem;
    margin: 1rem 0;
}
.ob-card {
    background: #0c0c0c;
    border: 1px solid #1a1a1a;
    border-radius: 5px;
    padding: 1.1rem 1.3rem;
}
.ob-card h4 {
    font-family: 'Space Grotesk', sans-serif;
    font-size: 0.68rem;
    font-weight: 600;
    letter-spacing: 0.16em;
    text-transform: uppercase;
    color: #a3e635;
    margin: 0 0 0.65rem;
}
.ob-card p, .ob-card li, .ob-card pre {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.7rem;
    color: #444;
    line-height: 1.75;
    margin: 0;
}
.ob-card ul { padding-left: 1rem; }
.ob-card code {
    color: #a3e635;
    background: transparent;
    font-size: 0.67rem;
}
.embed-tbl {
    width: 100%;
    border-collapse: collapse;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.67rem;
    margin-top: 0.5rem;
}
.embed-tbl th {
    text-align: left;
    color: #2a2a2a;
    font-size: 0.6rem;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.25rem 0.5rem;
    border-bottom: 1px solid #1a1a1a;
    font-weight: 400;
}
.embed-tbl td {
    color: #444;
    padding: 0.3rem 0.5rem;
    border-bottom: 1px solid #141414;
    vertical-align: middle;
}
.embed-tbl tr:last-child td { border:none; }
.embed-tbl td:first-child { color: #a3e635; }

/* ── Footer ── */
.footer {
    text-align: center;
    padding: 1.5rem 0 0.5rem;
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.6rem;
    color: #1e1e1e;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    border-top: 1px solid #141414;
    margin-top: 3rem;
}
.footer span { color: #2a2a2a; }

hr { border-color: #1a1a1a !important; margin: 1.25rem 0 !important; }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# REDESIGNED SIDEBAR
# ============================================================================

with st.sidebar:

    st.markdown("### Connection")
    ollama_url = st.text_input("Host URL", value=OLLAMA_BASE_URL, label_visibility="collapsed")
    CONFIG["ollama_url"] = ollama_url

    running   = check_ollama(ollama_url)
    installed = get_installed(ollama_url) if running else []

    if running:
        st.success(f"Ollama running · {len(installed)} model(s) installed")
        if installed:
            for m in installed:
                st.markdown(
                    f'<div class="kv"><span class="kv-k">installed</span>'
                    f'<span class="kv-v">{m}</span></div>',
                    unsafe_allow_html=True
                )
    else:
        st.error("Ollama unreachable — run: `ollama serve`")

    st.markdown("---")

    # Chat model
    st.markdown("### Chat Model")
    chat_list    = list(dict.fromkeys(installed + ALL_CHAT_MODELS))
    default_chat = chat_list.index(installed[0]) if installed else 0
    sel_chat     = st.selectbox("Model", chat_list, index=default_chat, label_visibility="collapsed")

    if sel_chat in installed:
        st.success(f"`{sel_chat}` ready")
    else:
        st.warning(f"`{sel_chat}` not installed")
        if st.button(f"Pull `{sel_chat}`", use_container_width=True):
            with st.status(f"Pulling `{sel_chat}`...", expanded=True) as s:
                try:
                    for u in pull_stream(sel_chat, ollama_url):
                        if u.get("status"): st.write(u["status"])
                    s.update(label="Done!", state="complete"); st.rerun()
                except Exception as e:
                    s.update(label=f"Error: {e}", state="error")

    CONFIG["llm_model"] = sel_chat

    st.markdown("---")

    # Embedding model
    st.markdown("### Embedding Model")

    auto_embed_model, auto_embed_mode = best_embed(installed)
    embed_opts, embed_labels = [], []

    for m in installed:
        embed_opts.append(("ollama", m))
        lbl = f"ollama · {m}"
        if any(em in m for em in KNOWN_EMBED_MODELS):
            lbl += " ★"
        embed_labels.append(lbl)

    embed_opts.append(("huggingface", "sentence-transformers/all-MiniLM-L6-v2"))
    embed_labels.append("huggingface · all-MiniLM-L6-v2 ✓")

    def_embed_idx = len(embed_opts) - 1
    for i, (mode, model) in enumerate(embed_opts):
        if model == auto_embed_model and mode == auto_embed_mode:
            def_embed_idx = i
            break

    sel_embed_idx = st.selectbox(
        "Embedding source",
        range(len(embed_labels)),
        format_func=lambda i: embed_labels[i],
        index=def_embed_idx,
        label_visibility="collapsed",
        help="★ = dedicated embed model (best). HuggingFace requires no extra pull.",
    )

    em_mode, em_model = embed_opts[sel_embed_idx]
    CONFIG["embedding_model"] = em_model
    CONFIG["embedding_mode"]  = em_mode

    if em_mode == "huggingface":
        st.info("HuggingFace embeddings — no Ollama pull required")
    else:
        st.success(f"`{em_model}` ready for embeddings")

    with st.expander("Pull a dedicated embed model"):
        for em in KNOWN_EMBED_MODELS:
            c1, c2 = st.columns([3, 1])
            c1.markdown(f'<span style="font-size:0.7rem;color:#555;">{em}</span>', unsafe_allow_html=True)
            mark = "✓" if any(em in m for m in installed) else "Pull"
            if c2.button(mark, key=f"pull_em_{em}"):
                with st.status(f"Pulling `{em}`...", expanded=True) as s:
                    try:
                        for u in pull_stream(em, ollama_url):
                            if u.get("status"): st.write(u["status"])
                        s.update(label="Done!", state="complete"); st.rerun()
                    except Exception as e:
                        s.update(label=f"Error: {e}", state="error")

    st.markdown("---")

    # Generation settings
    st.markdown("### Generation")
    CONFIG["temperature"] = st.slider("Temperature", 0.0, 1.0, CONFIG["temperature"], 0.1)
    CONFIG["max_tokens"]  = st.slider("Max Tokens",  256, 4096, CONFIG["max_tokens"],  256)
    CONFIG["retrieval_k"] = st.slider("Top-K Chunks", 1, 10,   CONFIG["retrieval_k"],  1)
    CONFIG["chunk_size"]  = st.slider("Chunk Size",  200, 2000, CONFIG["chunk_size"],   100)

    st.markdown("---")

    # Config dump
    st.markdown("### Config")
    for k, v in CONFIG.items():
        if k != "ollama_url":
            st.markdown(
                f'<div class="kv"><span class="kv-k">{k}</span>'
                f'<span class="kv-v">{v}</span></div>',
                unsafe_allow_html=True
            )
    st.markdown(
        '<div style="margin-top:1rem;padding:0.6rem;background:#060606;border:1px solid #141414;'
        'border-radius:3px;font-size:0.62rem;color:#2a2a2a;font-family:JetBrains Mono,monospace;'
        'letter-spacing:0.06em;">100% local · no API keys · data stays on device</div>',
        unsafe_allow_html=True
    )


# ============================================================================
# REDESIGNED MAIN UI
# ============================================================================

# Hero
st.markdown(
    '<div class="hero">'
    '<div class="hero-eyebrow">Local RAG · Ollama · 100% Offline</div>'
    '<div class="hero-title">Paracetamol<br><em>Research Assistant</em></div>'
    '<div class="hero-tagline">No API keys · No cloud · <span>Data stays on your machine</span></div>'
    '</div>',
    unsafe_allow_html=True
)

# Ollama gate
if not running:
    st.error("Ollama is not running. Start it with `ollama serve` then refresh.")
    st.stop()

# Upload
st.markdown('<div class="sec">01 — Document</div>', unsafe_allow_html=True)
uploaded_file = st.file_uploader("Drop your Paracetamol PDF here", type="pdf", label_visibility="visible")

if uploaded_file:
    cache_key = f"{uploaded_file.name}__{em_model}__{em_mode}"

    if st.session_state.get("cache_key") != cache_key:
        src_label = f"huggingface · {em_model}" if em_mode == "huggingface" else f"ollama · {em_model}"
        with st.spinner(f"Chunking · Embedding via {src_label}…"):
            try:
                chunks = process_document(uploaded_file)
                st.session_state.vector_db   = create_vector_db(chunks)
                st.session_state.cache_key   = cache_key
                st.session_state.chunk_count = len(chunks)
                st.success(f"Indexed {len(chunks)} chunks from **{uploaded_file.name}**  ·  embed: `{em_model}`")
            except Exception as e:
                st.error(f"Error: {e}")
                st.stop()
    else:
        st.success(
            f"**{uploaded_file.name}** — {st.session_state.chunk_count} chunks ready  ·  embed: `{em_model}`"
        )

    # Query
    st.markdown('<div class="sec">02 — Query</div>', unsafe_allow_html=True)
    query = st.text_input(
        "Ask a question",
        placeholder="e.g., What are the side effects of Paracetamol?",
        label_visibility="visible",
    )

    if query:
        with st.spinner(f"Running `{CONFIG['llm_model']}`…"):
            try:
                llm = OllamaLLM(
                    model_name=CONFIG["llm_model"],
                    temperature=CONFIG["temperature"],
                    max_tokens=CONFIG["max_tokens"],
                    base_url=CONFIG["ollama_url"],
                )
                retriever = st.session_state.vector_db.as_retriever(
                    search_kwargs={"k": CONFIG["retrieval_k"]}
                )
                prompt = ChatPromptTemplate.from_template("""You are an expert assistant for Paracetamol questions.

Instructions:
1. Use ONLY the provided context to answer
2. If not in context, say you don't know
3. Be accurate and concise

Context:
{context}

Question: {question}

Answer:""")
                chain = (
                    {"context": retriever | fmt, "question": RunnablePassthrough()}
                    | prompt | llm | StrOutputParser()
                )
                answer = chain.invoke(query)
                src    = retriever.invoke(query)

                # Answer
                st.markdown('<div class="sec">03 — Answer</div>', unsafe_allow_html=True)
                st.markdown(
                    f'<div class="model-badge">🦙 ollama · {CONFIG["llm_model"]}</div>',
                    unsafe_allow_html=True
                )
                st.markdown(
                    f'<div class="answer-wrap">'
                    f'<div class="answer-tag">Response</div>'
                    f'{answer}'
                    f'</div>',
                    unsafe_allow_html=True
                )

                # Sources
                st.markdown(
                    f'<div class="sec">04 — Source Chunks · top {CONFIG["retrieval_k"]}</div>',
                    unsafe_allow_html=True
                )
                for i, doc in enumerate(src, 1):
                    page = doc.metadata.get("page", "?")
                    with st.expander(f"chunk {i:02d}  ·  page {page}"):
                        st.write(doc.page_content)

                st.info(
                    f"{len(src)} chunks retrieved  ·  "
                    f"chat: `{CONFIG['llm_model']}`  ·  "
                    f"embed: `{CONFIG['embedding_model']}` ({CONFIG['embedding_mode']})"
                )

            except Exception as e:
                st.error(f"Error: {e}")
                with st.expander("Stack trace"):
                    import traceback; st.code(traceback.format_exc(), language="python")

    # Architecture
    with st.expander("System architecture"):
        st.markdown(
            '<div class="pipe">'
            '<div class="pipe-node"><strong>1 · Load</strong>PyPDFLoader parses PDF</div>'
            '<div class="pipe-node"><strong>2 · Chunk</strong>RecursiveCharacterTextSplitter</div>'
            '<div class="pipe-node"><strong>3 · Embed</strong>Ollama or HuggingFace</div>'
            '<div class="pipe-node"><strong>4 · Store</strong>FAISS vector index</div>'
            '<div class="pipe-node"><strong>5 · Retrieve</strong>Top-k cosine similarity</div>'
            '<div class="pipe-node"><strong>6 · Generate</strong>Ollama LLM (local)</div>'
            '<div class="pipe-node"><strong>7 · Output</strong>Answer + source display</div>'
            '</div>',
            unsafe_allow_html=True
        )
        c1, c2 = st.columns(2)
        with c1:
            for k, v in [
                ("chunk_size", CONFIG["chunk_size"]),
                ("chunk_overlap", CONFIG["chunk_overlap"]),
                ("embed_model", CONFIG["embedding_model"]),
                ("embed_mode", CONFIG["embedding_mode"]),
                ("vector_store", "FAISS"),
            ]:
                st.markdown(
                    f'<div class="kv"><span class="kv-k">{k}</span>'
                    f'<span class="kv-v">{v}</span></div>',
                    unsafe_allow_html=True
                )
        with c2:
            for k, v in [
                ("engine", "Ollama (local)"),
                ("llm_model", CONFIG["llm_model"]),
                ("temperature", CONFIG["temperature"]),
                ("max_tokens", CONFIG["max_tokens"]),
                ("retrieval_k", CONFIG["retrieval_k"]),
            ]:
                st.markdown(
                    f'<div class="kv"><span class="kv-k">{k}</span>'
                    f'<span class="kv-v">{v}</span></div>',
                    unsafe_allow_html=True
                )

else:
    # Onboarding state
    st.markdown('<div class="sec">Getting started</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="ob-grid">'

        '<div class="ob-card">'
        '<h4>Setup</h4>'
        '<ul>'
        '<li>Install Ollama: <code>ollama.com/install</code></li>'
        '<li>Start it: <code>ollama serve</code></li>'
        '<li>Pull a model: <code>ollama pull llama3</code></li>'
        '<li>Upload a PDF and ask away</li>'
        '</ul>'
        '</div>'

        '<div class="ob-card">'
        '<h4>How it works</h4>'
        '<ul>'
        '<li>PDF is chunked into overlapping pieces</li>'
        '<li>Chunks are embedded into vectors</li>'
        '<li>Your query finds the closest chunks</li>'
        '<li>Ollama generates a grounded answer</li>'
        '</ul>'
        '</div>'

        '<div class="ob-card" style="grid-column:span 2;">'
        '<h4>Embedding options</h4>'
        '<table class="embed-tbl">'
        '<thead><tr><th>Option</th><th>Pull required?</th><th>Quality</th></tr></thead>'
        '<tbody>'
        '<tr><td>HuggingFace all-MiniLM</td><td>No</td><td>Good — works out of the box</td></tr>'
        '<tr><td>llama3 (chat model)</td><td>Already installed</td><td>Good</td></tr>'
        '<tr><td>nomic-embed-text ★</td><td>Yes</td><td>Best — dedicated embed model</td></tr>'
        '<tr><td>mxbai-embed-large ★</td><td>Yes</td><td>Best — dedicated embed model</td></tr>'
        '</tbody>'
        '</table>'
        '</div>'

        '</div>',
        unsafe_allow_html=True
    )

# Footer
st.markdown(
    f'<div class="footer">'
    f'Ollama RAG · 100% Local · '
    f'chat: <span>{CONFIG["llm_model"]}</span> · '
    f'embed: <span>{CONFIG["embedding_model"]}</span>'
    f'</div>',
    unsafe_allow_html=True
)