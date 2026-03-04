# 💊 Paracetamol Document QA System
### Retrieval-Augmented Generation (RAG) · LangChain · Streamlit · Ollama / Groq

> An intelligent document question-answering system that ingests Paracetamol PDF documents, builds a semantic vector index, and generates grounded answers using a local or cloud LLM — with full source attribution.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Editions](#editions)
- [System Architecture](#system-architecture)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Running the App](#running-the-app)
- [Usage](#usage)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Troubleshooting](#troubleshooting)
- [Evaluation Checklist](#evaluation-checklist)

---

## Overview

The **Paracetamol Document QA System** allows users to upload any Paracetamol-related PDF and ask natural language questions about its content. It uses a full **RAG (Retrieval-Augmented Generation)** pipeline:

1. PDF is loaded and split into overlapping chunks
2. Chunks are embedded into dense vectors using `all-MiniLM-L6-v2`
3. Vectors are stored and indexed in a FAISS database
4. User queries retrieve the top-3 most semantically relevant chunks
5. A local or cloud LLM generates a grounded answer from the retrieved context
6. The answer and source chunks are displayed in a clean Streamlit UI

---

## Editions

| Feature | 🌐 Groq Edition (`app.py`) | 🦙 Ollama Edition (`paracetamol_ollama_rag.py`) |
|---|---|---|
| LLM Provider | Groq API (cloud) | Ollama (fully local) |
| API Key Required | ✅ Yes — free at console.groq.com | ❌ No |
| Internet Required | ✅ Yes | ❌ No |
| Data Privacy | Sent to Groq servers | Stays on your machine |
| Speed | Very fast | Depends on hardware |
| Cost | Free (Groq free tier) | Free |
| Best For | Quick setup, best performance | Privacy, offline use |

---

## System Architecture

```
┌──────────────────────────────────────────────────────────────┐
│                       RAG PIPELINE                           │
│                                                              │
│   PDF Upload  ──►  PyPDFLoader  ──►  TextSplitter           │
│                                      (1000 / 200 overlap)    │
│                                           │                  │
│                                           ▼                  │
│                              HuggingFace Embeddings          │
│                              (all-MiniLM-L6-v2, 384-dim)    │
│                                           │                  │
│                                           ▼                  │
│                              FAISS Vector Store              │
│                                           │                  │
│                    User Query ──► Embed Query                │
│                                           │                  │
│                                           ▼                  │
│                              Top-K Retrieval  (k=3)          │
│                                           │                  │
│                                           ▼                  │
│                         LLM (Ollama / Groq)                  │
│                                           │                  │
│                                           ▼                  │
│                    Streamlit UI ◄── Answer + Source Chunks   │
└──────────────────────────────────────────────────────────────┘
```

---

## Prerequisites

### Both Editions
- Python **3.9 or higher**
- pip

### 🌐 Groq Edition only
- Free Groq API key → [console.groq.com](https://console.groq.com) (no payment required)

### 🦙 Ollama Edition only
- Ollama installed → [ollama.com](https://ollama.com)
- At least one model pulled (e.g. `ollama pull llama3`)

---

## Installation

### Step 1 — Download the project

```bash
git clone https://github.com/your-username/paracetamol-doc-qa.git
cd paracetamol-doc-qa
```

Or download and extract the ZIP, then open a terminal in the project folder.

### Step 2 — Install Python dependencies

```bash
pip install -r requirements.txt
```

**Or install manually:**

```bash
pip install streamlit langchain langchain-community langchain-core \
            langchain-text-splitters faiss-cpu pypdf requests \
            python-dotenv sentence-transformers
```

> `sentence-transformers` downloads `all-MiniLM-L6-v2` (~90 MB) on first run.

### Step 3 — Ollama Edition only

**Install Ollama:**

```bash
# macOS / Linux
curl -fsSL https://ollama.com/install.sh | sh

# Windows — download from:
# https://ollama.com/download/windows
```

**Start Ollama:**

```bash
ollama serve
```

> If you see `bind: Only one usage of each socket address` — Ollama is already running. Skip this step.

**Pull a chat model (choose one):**

```bash
ollama pull llama3           # Recommended
ollama pull llama3.2         # Lighter and faster
ollama pull mistral          # Good alternative
ollama pull gemma2           # Google's model
```

**Pull a dedicated embedding model (optional — better quality):**

```bash
ollama pull nomic-embed-text      # Best quality ★
ollama pull mxbai-embed-large     # Also excellent ★
```

> If no embedding model is pulled, the app automatically falls back to the HuggingFace `all-MiniLM-L6-v2` model — no extra setup needed.

---

## Configuration

### 🌐 Groq Edition — `.env` file

Create a file named `.env` in the project root:

```env
GROQ_API_KEY=gsk_your_key_here

# Optional
OPENAI_API_KEY=sk_your_key_here
HF_TOKEN=hf_your_token_here
```

Get a **free** Groq API key at [console.groq.com](https://console.groq.com).

### 🦙 Ollama Edition — no configuration needed

The app auto-detects your installed models and connects to `http://localhost:11434` by default.

To use a custom Ollama host:

```bash
# Windows
set OLLAMA_HOST=http://your-host:11434

# macOS / Linux
export OLLAMA_HOST=http://your-host:11434
```

---

## Running the App

### 🌐 Groq Edition

```bash
streamlit run app.py
```

### 🦙 Ollama Edition

```bash
streamlit run paracetamol_ollama_rag.py
```

Open your browser at: **http://localhost:8501**

---

## Usage

| Step | Action |
|---|---|
| 1 | Upload a Paracetamol PDF using the file uploader |
| 2 | Wait for chunking and indexing to complete |
| 3 | Type a question in the query box |
| 4 | Read the generated answer |
| 5 | Expand source chunks to see the retrieved passages |
| 6 | Adjust model, temperature, chunk size in the sidebar |

**Example questions:**
- *What are the side effects of Paracetamol?*
- *What is the recommended adult dosage?*
- *What drug interactions are mentioned?*
- *Is Paracetamol safe during pregnancy?*
- *What is the mechanism of action of Paracetamol?*

---

## Project Structure

```
paracetamol-doc-qa/
│
├── app.py                          # Groq edition (cloud LLM)
├── paracetamol_ollama_rag.py       # Ollama edition (local LLM)
├── requirements.txt                # Python dependencies
├── .env                            # API keys — create this, do not commit
├── .gitignore
└── README.md
```

---

## Tech Stack

| Layer | Technology |
|---|---|
| UI Framework | Streamlit |
| Document Loader | LangChain · PyPDFLoader |
| Text Splitter | RecursiveCharacterTextSplitter |
| Embedding Model | HuggingFace `sentence-transformers/all-MiniLM-L6-v2` |
| Vector Database | FAISS (Facebook AI Similarity Search) |
| LLM — Local | Ollama (Llama 3, Mistral, Gemma 2, Phi 4, etc.) |
| LLM — Cloud | Groq API (Llama 3.3 70B, Gemma 2, etc.) |
| Chain Orchestration | LangChain LCEL (Runnable pipelines) |

### Available Models

**Groq (cloud · free tier):**

| Model | Context Window | Notes |
|---|---|---|
| `llama-3.3-70b-versatile` | 128K | Best quality — default |
| `llama-3.1-8b-instant` | 128K | Fastest responses |
| `llama3-70b-8192` | 8K | High quality |
| `gemma2-9b-it` | 8K | Google Gemma 2 |
| `compound-beta` | — | Groq compound model |

**Ollama (local · free):**

| Model | Pull Command | Notes |
|---|---|---|
| `llama3` | `ollama pull llama3` | Recommended |
| `llama3.2` | `ollama pull llama3.2` | Lighter / faster |
| `mistral` | `ollama pull mistral` | Strong alternative |
| `gemma2` | `ollama pull gemma2` | Google's model |
| `phi4` | `ollama pull phi4` | Microsoft's model |

**Embedding models (Ollama):**

| Model | Pull Command | Notes |
|---|---|---|
| `nomic-embed-text` | `ollama pull nomic-embed-text` | Best quality ★ |
| `mxbai-embed-large` | `ollama pull mxbai-embed-large` | Excellent ★ |
| `all-minilm` | `ollama pull all-minilm` | Lightweight |
| HuggingFace fallback | No pull needed | Works out of the box ✓ |

---

## Troubleshooting

| Error | Cause | Fix |
|---|---|---|
| `ModuleNotFoundError: No module named 'langchain.llms'` | Outdated LangChain | Change import to `from langchain_core.language_models.llms import LLM` |
| `File does not exist: app.py` | Wrong directory | `cd` into the project folder first |
| `bind: Only one usage of each socket address` | Ollama already running | Skip `ollama serve` — it is already active |
| `Model not found` | Model not pulled | Run `ollama pull llama3` |
| `Cannot connect to Ollama` | Ollama not running | Run `ollama serve` |
| `GROQ_API_KEY not found` | Missing `.env` file | Create `.env` with your Groq key |
| Blank / slow first run | Downloading embedding model | Wait ~1 min for `all-MiniLM-L6-v2` to download |

**Verify your setup before running:**

```bash
python --version                  # Must be 3.9+
streamlit --version               # Must be installed
ollama list                       # Ollama edition: must show at least one model
curl http://localhost:11434       # Ollama edition: must return "Ollama is running"
```

---

## Evaluation Checklist

| Requirement | Status |
|---|---|
| ✅ Load PDF document | `PyPDFLoader` |
| ✅ RecursiveCharacterTextSplitter | chunk_size=1000, overlap=200 |
| ✅ HuggingFace Embeddings | `all-MiniLM-L6-v2` |
| ✅ FAISS vector store | In-memory, normalised embeddings |
| ✅ Top-3 chunk retrieval | Configurable via sidebar (1–10) |
| ✅ RetrievalQA chain | LangChain LCEL pipeline |
| ✅ Source documents returned | Page number + text displayed |
| ✅ Streamlit file uploader | PDF only |
| ✅ Question input box | With placeholder example |
| ✅ Answer display | Styled answer card |
| ✅ Retrieved chunks display | Expandable per chunk |
| ✅ Loading spinner | On indexing and generation |

---

<div align="center">

Built with LangChain · FAISS · Streamlit · Ollama · Groq

**Submitted for:** Document Question Answering System using RAG · March 2026

</div>
=======
# Paracetamol-Document-QA-System
>>>>>>> 034976f46e820537e7205bf6ddba95bbf869c95e
