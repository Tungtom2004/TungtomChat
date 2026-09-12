<div align="center">

# 🤖 TungtomChat

### A Retrieval-Augmented Generation (RAG) Chatbot for PTIT Academic Content

[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?logo=python&logoColor=white)](https://python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.35+-FF4B4B?logo=streamlit&logoColor=white)](https://streamlit.io)
[![LangChain](https://img.shields.io/badge/LangChain-0.2+-1C3C3C?logo=langchain&logoColor=white)](https://langchain.com)
[![ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-FF6B35)](https://trychroma.com)
[![Azure OpenAI](https://img.shields.io/badge/Azure_OpenAI-GPT--4o-0078D4?logo=microsoft-azure&logoColor=white)](https://azure.microsoft.com/en-us/products/ai-services/openai-service)

*An intelligent academic assistant that answers questions about PTIT university courses using lecture slides, semantic routing, reranking, and multimodal reasoning.*

</div>

---

## 📋 Table of Contents

- [Overview](#-overview)
- [Key Features](#-key-features)
- [System Architecture](#-system-architecture)
- [Project Structure](#-project-structure)
- [Technology Stack](#-technology-stack)
- [Knowledge Base](#-knowledge-base)
- [API Reference](#-api-reference)
- [Core Components](#-core-components)
- [Data Pipeline](#-data-pipeline)
- [RAG Evaluation](#-rag-evaluation)
- [Configuration](#-configuration)
- [Installation & Setup](#-installation--setup)
- [Running the Application](#-running-the-application)
- [Environment Variables](#-environment-variables)
- [Troubleshooting](#-troubleshooting)

---

## 🌟 Overview

**TungtomChat** is an intelligent chatbot built for students at **Posts and Telecommunications Institute of Technology (PTIT)**. It answers academic questions by retrieving relevant content from PTIT lecture slides, combining:

- **Retrieval-Augmented Generation (RAG)** — grounding answers in actual course materials
- **Semantic Routing** — intelligently directing queries to the correct subject module
- **Cross-Encoder Reranking** — surfacing the most relevant passages before answering
- **Query Reflection** — rewriting follow-up questions into standalone queries for better retrieval
- **Tool Calling** — fetching real-time data (time, Google Search) for general queries
- **Image Reasoning** — solving visual/mathematical problems from uploaded images

The system is deployed as a **Streamlit frontend** communicating with a **FastAPI backend** — a clean, production-style separation of concerns.

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 📚 **Multi-subject RAG** | Answers questions across 5 PTIT subjects from vectorized lecture slides |
| 🔀 **Semantic Router** | Embeddings-based cosine similarity routing to the correct subject knowledge base |
| 🔁 **Query Reflection** | LLM rewrites follow-up questions into standalone queries using conversation history |
| 🏆 **Cross-Encoder Reranking** | `Alibaba-NLP/gte-multilingual-reranker-base` reranks candidate passages |
| 🧮 **Compute Agent** | Dedicated reasoning mode for CPU scheduling and image processing calculations |
| 🌐 **Chitchat + Tools** | General conversation with LangChain tool calling (real-time clock, Google Search) |
| 🖼️ **Image Chat** | Upload an image and ask questions — GPT-4o solves it step-by-step |
| 🌍 **Multilingual Output** | Responds in 12 languages (auto-detect or user-selected) |
| ⚡ **Streaming** | LLM responses are streamed token-by-token for low perceived latency |
| 📊 **RAG Evaluation** | Automated evaluation pipeline with Correctness, Faithfulness, and Relevancy metrics |

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                             USER INTERFACE                                  │
│                    Streamlit Frontend (app.py)                              │
│         Chat Input │ Image Upload │ History Display │ Language Picker       │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │ HTTP POST  /chat  or  /chat_image
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         FastAPI Backend (api/main.py)                       │
│              POST /chat ─────────► chat_engine()                            │
│              POST /chat_image ───► chat_engine_image()                      │
│              GET  /health ───────► {"status": "ok"}                         │
└────────────────────────────┬────────────────────────────────────────────────┘
                             │
                             ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                         CORE ENGINE (core/engine.py)                        │
│                                                                             │
│  ┌──────────────────┐                                                       │
│  │  Query Reflection│  ← Rewrites follow-up questions into standalone        │
│  │ (core/reflection)│    Vietnamese queries using conversation history       │
│  └────────┬─────────┘                                                       │
│           │                                                                 │
│           ▼                                                                 │
│  ┌──────────────────────────────────────────────────────┐                  │
│  │               Semantic Router (core/router.py)        │                  │
│  │                                                       │                  │
│  │  Embeddings ──► Cosine Similarity ──► Route Label     │                  │
│  │                                                       │                  │
│  │  ┌────────────┐  ┌──────────────────┐  ┌──────────┐  │                  │
│  │  │ hedieuhanh │  │ Phantichthietkeht│  │ tutuonghcm│  │                  │
│  │  │ (OS)       │  │ tt (ITSA)        │  │ (HCM Thgt)│  │                  │
│  │  └────────────┘  └──────────────────┘  └──────────┘  │                  │
│  │  ┌────────────┐  ┌──────────────────┐  ┌──────────┐  │                  │
│  │  │  Xulyanh   │  │   Laptrinhdidong │  │ chitchat │  │                  │
│  │  │ (Img Proc) │  │   (Mobile Dev)   │  │ (General)│  │                  │
│  │  └────────────┘  └──────────────────┘  └──────────┘  │                  │
│  └───────────────────────────┬──────────────────────────┘                  │
│                              │                                              │
│           ┌──────────────────┼──────────────────────┐                      │
│           ▼                  ▼                       ▼                      │
│  ┌─────────────┐   ┌─────────────────┐   ┌─────────────────┐               │
│  │ RAG Pipeline│   │ Compute Agent   │   │ Chitchat + Tools│               │
│  │             │   │                 │   │                 │               │
│  │ Retrieve    │   │ Retrieve + ctx  │   │ Tool Calling    │               │
│  │ Rerank      │   │ Structured calc │   │ - get_time      │               │
│  │ Build ctx   │   │ step-by-step    │   │ - google_search │               │
│  │ LLM answer  │   │ answer          │   │ LLM answer      │               │
│  └──────┬──────┘   └────────┬────────┘   └────────┬────────┘               │
│         │                   │                      │                        │
└─────────┼───────────────────┼──────────────────────┼────────────────────────┘
          │                   │                      │
          ▼                   ▼                      ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│                          KNOWLEDGE & MODEL LAYER                            │
│                                                                             │
│  ChromaDB (.chroma_db)          Azure OpenAI GPT-4o                        │
│  └── ptit_corpus collection     └── Chat Completions API                    │
│      └── child_chunks (vector)                                              │
│                                                                             │
│  Embedding Model                Cross-Encoder Reranker                     │
│  └── Alibaba-NLP/               └── Alibaba-NLP/                           │
│      gte-multilingual-base          gte-multilingual-reranker-base          │
└─────────────────────────────────────────────────────────────────────────────┘
```

### Request Flow (Text Query)

```
User message
    │
    ├─► [Reflection] Rewrite into standalone question
    │
    ├─► [Router] Cosine similarity → subject label + confidence score
    │
    ├─► [RAG] if subject is known or unknown:
    │       ├── retrieve() → top-30 candidates from ChromaDB
    │       ├── simple_rerank() → CrossEncoder or heuristic → top-10
    │       ├── build_context_guarded() → filtered context string
    │       └── LLM (streaming) → final answer with inline citations
    │
    ├─► [Compute Agent] if subject ∈ {hedieuhanh, xulyanh} AND numeric keywords:
    │       └── Structured calculation with formulas, steps, result
    │
    └─► [Chitchat] if route == "chitchat":
            ├── LLM with bound tools
            ├── Tool invocations (time / Google Search)
            └── Final answer
```

---

## 📁 Project Structure

```
TungtomChat/
│
├── 🌐 app.py                       # Streamlit frontend UI
├── 🚀 api/
│   ├── __init__.py
│   └── main.py                     # FastAPI backend — /chat, /chat_image, /health
│
├── ⚙️  core/
│   ├── __init__.py
│   ├── engine.py                   # Main chat orchestration logic
│   ├── router.py                   # Semantic routing (embeddings + cosine similarity)
│   ├── rag.py                      # Retrieve, rerank, build context
│   ├── reflection.py               # Query rewriting with chat history
│   ├── tools.py                    # LangChain tools: get_time, google_search
│   └── loaders.py                  # Singleton loaders: LLM, embedder, collection, reranker
│
├── 🔧 config.py                    # Global config: DB paths, retrieval params, LLM, language options
├── 📐 embeddings.py                # Embedding model factory (gte-multilingual-base)
├── 🏆 rerank.py                    # CrossEncoder reranker wrapper
├── 💬 reflection.py                # Legacy reflection class (early prototype)
│
├── 🗃️  data/
│   ├── parent_docs.jsonl           # Full section texts (~779 KB)
│   └── child_chunks.jsonl          # Overlapping text chunks for retrieval (~1 MB)
│
├── 📂 Slides_PTIT/                 # Source PDF lecture slides
│   ├── Laptrinhdidong.pdf          # Mobile Programming
│   ├── NHDT.pdf                    # Computer Networks
│   ├── Phantichthietkehttt.pdf     # IT Systems Analysis & Design
│   ├── Xulyanh.pdf                 # Image Processing
│   ├── hedieuhanh.pdf              # Operating Systems
│   └── tutuonghcm.pdf             # Ho Chi Minh Thought
│
├── 🗄️  .chroma_db/                 # Persisted ChromaDB vector store
│   └── ptit_corpus/                # Collection of embedded child chunks
│
├── 📊 MinerU_out/                  # Structured JSON output from MinerU PDF parser
├── 🧪 evaluate_rag.py              # RAG evaluation pipeline (LlamaIndex + 3 metrics)
├── 📈 rag_evaluation_results_2.json # Saved evaluation results
├── 📋 tv.json                      # Evaluation test set (question-answer pairs)
│
├── 🔨 split.py                     # PDF → structured chunks (MinerU JSON parser)
├── 📥 chunks.py                    # Utility: load child chunks from JSONL
├── 🔢 index.py                     # Utility: normalize subject field in JSONL
├── 🗂️  vectorstore.py              # Legacy: build ChromaDB from LangChain documents
├── 🧪 test.py                      # Manual testing / experiments
├── 🎯 demo.py                      # Quick demo script
│
└── 📦 requirements.txt             # Python dependencies
```

---

## 🛠️ Technology Stack

### Backend & Orchestration

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **API Server** | FastAPI 0.111+ | RESTful backend, Pydantic schemas |
| **LLM Orchestration** | LangChain 0.2+ | Message formatting, tool binding, streaming |
| **LLM** | Azure OpenAI GPT-4o | Answer generation, query reflection, tool calls |
| **Evaluation** | LlamaIndex | Correctness, Faithfulness, Relevancy metrics |

### Retrieval & Embeddings

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **Vector Store** | ChromaDB (persistent) | Stores and queries embedded text chunks |
| **Embedding Model** | `Alibaba-NLP/gte-multilingual-base` (768-dim) | Multilingual dense embeddings |
| **Reranker** | `Alibaba-NLP/gte-multilingual-reranker-base` | Cross-Encoder passage reranking |
| **Routing** | Cosine similarity on sample embeddings | Subject-level semantic routing |

### Frontend

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **UI** | Streamlit | Chat interface, image upload, session state |
| **HTTP Client** | `requests` | Calls FastAPI backend |

### Data Processing

| Component | Technology | Purpose |
|-----------|-----------|---------|
| **PDF Parser** | MinerU | Extracts structured JSON (blocks, titles, tables) from PDFs |
| **Text Splitting** | Custom (`split.py`) | Hierarchical parent-child chunking with overlap |
| **Normalization** | `unicodedata` + regex | Vietnamese text normalization |

### External Services

| Service | Usage |
|---------|-------|
| **Azure OpenAI** | GPT-4o for all LLM calls |
| **SerpAPI** | Google Search results for chitchat tool |

---

## 📚 Knowledge Base

TungtomChat answers questions across **6 PTIT subjects** sourced from official lecture slide PDFs:

| Subject Key | PDF File | Topic |
|-------------|---------|-------|
| `hedieuhanh` | `hedieuhanh.pdf` | Operating Systems — processes, scheduling, memory, deadlocks |
| `Phantichthietkehttt` | `Phantichthietkehttt.pdf` | IT Systems Analysis & Design — UML, use cases, requirements |
| `tutuonghcm` | `tutuonghcm.pdf` | Ho Chi Minh Thought — ideology, history, philosophy |
| `Xulyanh` | `Xulyanh.pdf` | Image Processing — filters, segmentation, morphology |
| `Laptrinhdidong` | `Laptrinhdidong.pdf` | Mobile Programming |
| `NHDT` | `NHDT.pdf` | Computer Networks |

### Document Processing Pipeline

```
PDF Files (Slides_PTIT/)
        │
        ▼
   MinerU Parser
   (structured JSON with blocks, titles, tables)
        │
        ▼
   split.py — process_middle_json()
   ┌────────────────────────────────┐
   │ Hierarchical section parsing   │
   │ - title blocks → section keys  │
   │ - text blocks → buffered text  │
   │ - table blocks → Markdown rows │
   └────────────────────────────────┘
        │
        ├──► parent_docs.jsonl   (full section text, ~800+ chars)
        │    {parent_id, subject, section, title_path, text, word_count}
        │
        └──► child_chunks.jsonl  (overlapping chunks, 1100 chars, 180 overlap)
             {parent_id, subject, section, content, order_start, order_end}
                    │
                    ▼
             ChromaDB (.chroma_db)
             collection: ptit_corpus / ptit_giaotrinh_2
             embedding: gte-multilingual-base (768-dim)
```

### Chunking Strategy

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `chunk_size` | 1,100 chars | Balances context richness vs. retrieval precision |
| `chunk_overlap` | 180 chars | Prevents information loss at boundaries |
| `min_section_len` | 50 chars | Filters out very short, uninformative sections |
| Sentence-aware split | `rfind('.')` | Splits at sentence boundaries where possible |

---

## 🔌 API Reference

### Base URL

```
http://localhost:8000
```

### Endpoints

#### `POST /chat` — Text Chat

**Request Body:**
```json
{
  "message": "Giải thuật Round Robin là gì?",
  "history": [
    {"role": "user", "content": "Hệ điều hành có mấy chương?"},
    {"role": "assistant", "content": "Môn học có 8 chương..."}
  ],
  "debug": false
}
```

**Response:**
```json
{
  "answer": "Round Robin là giải thuật lập lịch CPU...[hedieuhanh/Lập lịch CPU]",
  "route": "hedieuhanh",
  "confidence": 0.82,
  "logs": null
}
```

**Response Fields:**

| Field | Type | Description |
|-------|------|-------------|
| `answer` | `string` | LLM-generated answer with inline citations like `[subject/section]` |
| `route` | `string` | Detected subject route (`hedieuhanh`, `Xulyanh`, `chitchat`, `unknown`, ...) |
| `confidence` | `float` | Max cosine similarity score from the semantic router (0.0 – 1.0) |
| `logs` | `list[str] \| null` | Tool call logs when `debug=true` and route is `chitchat` |

---

#### `POST /chat_image` — Image Chat

**Request Body:**
```json
{
  "message": "Tính toán convolution của ma trận này",
  "image_base64": "<base64-encoded-image-string>",
  "image_ext": "png"
}
```

**Response:**
```json
{
  "answer": "Step 1: Given the kernel matrix...\nStep 2: Apply convolution...",
  "route": "image"
}
```

---

#### `GET /health` — Health Check

**Response:**
```json
{"status": "ok"}
```

---

## 🔧 Core Components

### 1. Semantic Router (`core/router.py`)

Routes queries to the correct subject by comparing the query embedding against a pre-indexed set of sample questions per subject using **cosine similarity**.

```python
# Example routing decision
query = "Deadlock trong hệ điều hành là gì?"
route, confidence = route_semantic(query)
# → route="hedieuhanh", confidence=0.87
```

**How it works:**
1. At startup, embed all sample questions for each route → build a matrix per subject
2. For each query: embed query → compute max cosine similarity with each matrix
3. Select the subject with the highest score
4. If `max_score < ROUTE_MIN_COS` (default: 0.30) → label as `"unknown"` and use full RAG

**Supported Routes:**

| Route | Subject |
|-------|---------|
| `hedieuhanh` | Operating Systems |
| `Phantichthietkehttt` | IT Systems Analysis & Design |
| `tutuonghcm` | Ho Chi Minh Thought |
| `Xulyanh` | Image Processing |
| `chitchat` | General conversation / tools |

---

### 2. Query Reflection (`core/reflection.py`)

Uses the LLM to rewrite follow-up questions into **standalone questions** that are self-contained without needing the conversation history.

```
History:
  user: "Hệ điều hành có mấy chương?"
  assistant: "Có 8 chương..."

Follow-up: "Chương 3 nói về gì?"

Rewritten: "Chương 3 của môn Hệ điều hành PTIT nói về gì?"
```

This dramatically improves retrieval quality for multi-turn conversations.

---

### 3. RAG Pipeline (`core/rag.py`)

Three-stage retrieval pipeline:

#### Stage 1 — Retrieve
```python
retrieve(collection, embedder, query, subject_hint=route, k=30)
```
- Embeds the query with `gte-multilingual-base`
- Queries ChromaDB with optional `subject` metadata filter
- Returns top-30 `(document, metadata, distance)` tuples

#### Stage 2 — Rerank
```python
simple_rerank(query, candidates, top_k=10)
```
- **Primary**: `CrossEncoder` (`gte-multilingual-reranker-base`) scores all 30 passages → sorts by score
- **Fallback**: Heuristic scoring — `0.7 × keyword_overlap + 0.3 × (1 / (1 + distance))`

#### Stage 3 — Build Context
```python
build_context_guarded(ranked, max_chars=900)
```
- Filters out passages with `distance > DIST_THRES` (default: 1.60)
- Formats each passage as: `[subject/section]\n{text[:900]}`
- Returns a single concatenated context string

---

### 4. Compute Agent

Activated when the route is `hedieuhanh` or `Xulyanh` **and** the query contains numeric computation keywords (e.g., `waiting time`, `burst`, `convolution`, `3x3`).

Provides structured answers in the format:
1. **Input Data** — list all given parameters
2. **Formula / Algorithm** — state the method
3. **Step-by-step Calculation** — show all work
4. **Result** — final answer

---

### 5. Tool Calling (`core/tools.py`)

The `chitchat` route uses LangChain's `bind_tools()` for LLM-native tool calling:

| Tool | Function | Trigger |
|------|---------|---------|
| `get_current_time` | Returns current time in Vietnamese for any timezone | "Bây giờ là mấy giờ?" |
| `google_search` | SerpAPI top-3 snippet search | General knowledge queries |

---

### 6. Singleton Loaders (`core/loaders.py`)

All expensive objects are lazy-loaded and cached as module-level singletons:

```python
load_llm()        # → AzureChatOpenAI (initialized once)
load_embedder()   # → HuggingFaceEmbeddings (initialized once)
load_collection() # → ChromaDB Collection (initialized once)
load_reranker()   # → CrossEncoder Reranker (initialized once, optional)
```

This avoids repeated model loading on each API request.

---

## 🔨 Data Pipeline

### Step 1 — Parse PDFs with MinerU

```bash
# MinerU produces structured JSON in MinerU_out/
# Each subfolder corresponds to a subject
```

MinerU extracts PDF content as structured blocks:
- `title` blocks → section headings
- `text` blocks → paragraph content
- `table` blocks → tabular data (formatted as `| col1 | col2 |`)

### Step 2 — Generate Chunks

```bash
python split.py
```

Reads `MinerU_out/` and writes:
- `data/parent_docs.jsonl` — one record per section
- `data/child_chunks.jsonl` — overlapping chunks ready for embedding

### Step 3 — Index into ChromaDB

```bash
python vectorstore.py
```

Reads `data/child_chunks.jsonl`, embeds all chunks with `gte-multilingual-base`, and persists to `.chroma_db/`.

---

## 📊 RAG Evaluation

`evaluate_rag.py` runs a comprehensive evaluation using a hand-crafted test set (`tv.json`):

### Metrics (via LlamaIndex Evaluators)

| Metric | Description | Evaluator |
|--------|-------------|-----------|
| **Correctness** | Is the answer factually correct vs. reference? | `CorrectnessEvaluator` (LLM-as-judge) |
| **Faithfulness** | Is the answer grounded in the retrieved context? | `FaithfulnessEvaluator` |
| **Relevancy** | Is the retrieved context relevant to the question? | `RelevancyEvaluator` |

### Evaluation Pipeline

```
tv.json (question-answer pairs)
        │
        ▼
LlamaIndex VectorStoreIndex → query_engine.query(question)
        │
        ├── RAG answer
        └── source_nodes (raw contexts)
                │
                ▼
        CrossEncoder Reranker → top-3 reranked contexts
                │
                ▼
        LLM Evaluators (Correctness / Faithfulness / Relevancy)
                │
                ▼
        rag_evaluation_results_2.json
```

### Running Evaluation

```bash
python evaluate_rag.py
# Results saved to: rag_evaluation_results_2.json
```

---

## ⚙️ Configuration

### `config.py` — Global Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DB_PATH` | `.chroma_db` | ChromaDB persistence directory |
| `COLLECTION_NAME` | `ptit_corpus` | ChromaDB collection name |
| `FETCH_K` | `30` | Number of candidates retrieved from ChromaDB |
| `TOP_K` | `10` | Number of passages kept after reranking |
| `MAX_CHARS_PER_DOC` | `800` | Max characters per context block |
| `RERANKER_MODEL` | `Alibaba-NLP/gte-multilingual-reranker-base` | Cross-Encoder model |

### `core/rag.py` — Retrieval Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `DIST_THRES` | `1.60` | Max L2 distance to include a passage in context |

### `core/router.py` — Routing Settings

| Parameter | Default | Description |
|-----------|---------|-------------|
| `ROUTE_MIN_COS` | `0.30` | Minimum cosine similarity to assign a subject route |

All parameters can be overridden via environment variables.

---

## 📦 Installation & Setup

### Prerequisites

- Python >= 3.10
- Git

### 1. Clone the repository

```bash
git clone https://github.com/Tungtom2004/TungtomChat.git
cd TungtomChat
```

### 2. Create a virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Configure environment variables

Create a `.env` file in the project root:

```ini
# ── Azure OpenAI ──────────────────────────────────────────
AZURE_OPENAI_KEY=your_azure_openai_api_key
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT=gpt-4o
AZURE_OPENAI_VERSION=2024-02-01

# ── ChromaDB ──────────────────────────────────────────────
CHROMA_DB_PATH=.chroma_db
CHROMA_COLLECTION=ptit_corpus

# ── Retrieval Parameters ──────────────────────────────────
FETCH_K=30
TOP_K=10
DIST_THRES=1.60
ROUTE_MIN_COS=0.30

# ── Reranker ──────────────────────────────────────────────
RERANKER_MODEL=Alibaba-NLP/gte-multilingual-reranker-base

# ── Tools ─────────────────────────────────────────────────
SERPAPI_API_KEY=your_serpapi_key_here
```

### 5. (Optional) Rebuild the vector index

If you want to re-process the PDF slides from scratch:

```bash
# Step 1: Parse PDFs with MinerU (requires separate MinerU installation)
# → produces MinerU_out/{subject}/auto/*_middle.json

# Step 2: Generate chunks
python split.py

# Step 3: Index into ChromaDB
python vectorstore.py
```

> **Note:** A pre-built ChromaDB index is included in `.chroma_db/` and pre-processed chunks are in `data/`. You can skip this step if you just want to run the chatbot.

---

## 🚀 Running the Application

The application requires **two processes** running concurrently: the FastAPI backend and the Streamlit frontend.

### Terminal 1 — Start FastAPI Backend

```bash
uvicorn api.main:app --host 0.0.0.0 --port 8000 --reload
```

The backend will be available at: `http://localhost:8000`

Interactive API docs: `http://localhost:8000/docs`

### Terminal 2 — Start Streamlit Frontend

```bash
streamlit run app.py
```

The chatbot UI will open in your browser at: `http://localhost:8501`

---

## 🌍 Environment Variables

| Variable | Required | Default | Description |
|----------|:--------:|---------|-------------|
| `AZURE_OPENAI_KEY` | ✅ | — | Azure OpenAI API key |
| `AZURE_OPENAI_ENDPOINT` | ✅ | — | Azure resource endpoint URL |
| `AZURE_OPENAI_DEPLOYMENT` | ✅ | — | Deployment name (e.g., `gpt-4o`) |
| `AZURE_OPENAI_VERSION` | ✅ | — | API version (e.g., `2024-02-01`) |
| `CHROMA_DB_PATH` | ❌ | `.chroma_db` | Path to ChromaDB persistence directory |
| `CHROMA_COLLECTION` | ❌ | `ptit_corpus` | ChromaDB collection name |
| `FETCH_K` | ❌ | `30` | Candidates to fetch from vector store |
| `TOP_K` | ❌ | `10` | Passages to keep after reranking |
| `DIST_THRES` | ❌ | `1.60` | Distance threshold for context filtering |
| `ROUTE_MIN_COS` | ❌ | `0.30` | Minimum cosine score for subject routing |
| `RERANKER_MODEL` | ❌ | `Alibaba-NLP/gte-multilingual-reranker-base` | HuggingFace reranker model |
| `SERPAPI_API_KEY` | ❌ | — | SerpAPI key for Google Search tool |

---

## 🔍 Troubleshooting

### ❌ `Collection not found` error on startup

```
chromadb.errors.NotFoundError: Collection ptit_corpus does not exist.
```

**Solution:** The `.chroma_db` directory is either missing or the collection name is mismatched. Re-run the indexing step:
```bash
python vectorstore.py
```
Or set `CHROMA_COLLECTION` to match your existing collection name.

---

### ❌ Backend connection error in Streamlit

```
Không kết nối được backend: Connection refused
```

**Solution:** Make sure the FastAPI backend is running on port 8000:
```bash
uvicorn api.main:app --port 8000
```

---

### ❌ Azure OpenAI authentication error

```
openai.AuthenticationError: 401 Unauthorized
```

**Solution:** Verify your `.env` file contains the correct values for `AZURE_OPENAI_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`, and `AZURE_OPENAI_VERSION`.

---

### ❌ Reranker model download fails

```
OSError: Can't load tokenizer for 'Alibaba-NLP/gte-multilingual-reranker-base'
```

**Solution:** The model will be auto-downloaded from HuggingFace Hub on first use (~600 MB). Ensure you have internet access and sufficient disk space. If the reranker fails, the system falls back to heuristic scoring automatically.

---

### ❌ All answers cite `[unknown/]` sections

This indicates context was retrieved but metadata is incomplete. Re-index with correct `subject` and `section` fields in `child_chunks.jsonl`.

---

### ❌ Router always returns `"unknown"`

Increase `FETCH_K` or lower `ROUTE_MIN_COS` in your `.env`:
```ini
ROUTE_MIN_COS=0.20
```

---

## 📄 License

This project is developed for academic and research purposes at **Posts and Telecommunications Institute of Technology (PTIT)**.

---

<div align="center">

**Built with ❤️ by Tungtom**

*Powered by Azure OpenAI · LangChain · ChromaDB · Streamlit · FastAPI*

</div>