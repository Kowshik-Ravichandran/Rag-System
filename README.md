# 📚 Local RAG System

A fully local **Retrieval-Augmented Generation (RAG)** system for document question-answering. Upload PDF documents and ask questions — the AI answers **strictly** using content from your uploaded files.

**100% offline. No paid APIs. No cloud services. Runs entirely on your MacBook.**

---

## ✨ Features

- 📄 **PDF Upload** — Upload one or more PDF documents
- 🔍 **Dense Retrieval** — FAISS-powered semantic search finds the most relevant passages
- 🤖 **Local LLM** — Llama 3 (8B) via Ollama runs entirely on Apple Silicon
- 💬 **Chat Interface** — Streamlit-based conversational UI with streaming responses
- 📑 **Source Attribution** — See exactly which document chunks the AI used
- 📊 **Evaluation Dashboard** — Track response times, compare chunk sizes, precision\@k
- 🔒 **Privacy First** — Your documents never leave your machine

---

## 🏗️ Architecture

```
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   PDF Upload │────▶│  Text Extract │────▶│   Chunking   │
│  (Streamlit) │     │  (PyMuPDF)   │     │ (Token-based)│
└──────────────┘     └──────────────┘     └──────┬───────┘
                                                  │
                                                  ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────┐
│   Response   │◀────│  Llama 3 LLM │◀────│   Retrieval  │
│  (Streamlit) │     │  (Ollama)    │     │   (FAISS)    │
└──────────────┘     └──────────────┘     └──────────────┘
                                                  ▲
                                                  │
                                          ┌──────────────┐
                                          │  Embeddings  │
                                          │(MiniLM-L6-v2)│
                                          └──────────────┘
```

---

## 📁 Project Structure

```
Local RAG System/
├── app.py              # Streamlit chat interface & evaluation dashboard
├── ingest.py           # PDF extraction, chunking, embedding, FAISS indexing
├── retriever.py        # Dense retrieval using FAISS similarity search
├── generator.py        # Prompt construction & Ollama LLM interaction
├── utils.py            # Helpers: logging, token counting, precision@k
├── requirements.txt    # Python dependencies
├── README.md           # This file
├── vector_store/       # (auto-created) Saved FAISS index & metadata
└── logs/               # (auto-created) Query logs for evaluation
```

---

## 🚀 Setup Instructions

### Prerequisites

- **macOS** with Apple Silicon (M1/M2/M3/M4)
- **Python 3.10+**
- **Homebrew** (optional, for installing Ollama)

### Step 1: Install Ollama

Ollama runs large language models locally on your Mac.

```bash
# Option A: Download from the website
# Visit https://ollama.com and download the macOS app

# Option B: Install via Homebrew
brew install ollama
```

### Step 2: Pull the Llama 3 Model

```bash
# Start Ollama (if not already running)
ollama serve

# In a new terminal, pull the Llama 3 model
ollama pull llama3
```

> **Note:** The `llama3` model is ~4.7GB. It will download once and be cached locally.
> This is the 8B instruct variant, optimized for Apple Silicon.

### Step 3: Create a Virtual Environment

```bash
# Navigate to the project directory
cd "/Users/kowshik/Documents/Local RAG System"

# Create a virtual environment
python3 -m venv venv

# Activate it
source venv/bin/activate
```

### Step 4: Install Dependencies

```bash
pip install -r requirements.txt
```

> **First run note:** `sentence-transformers` will download the `all-MiniLM-L6-v2` model (~80MB) on first use. This happens once and is cached locally.

### Step 5: Run the Application

```bash
# Make sure Ollama is running in another terminal:
# ollama serve

# Start the Streamlit app
streamlit run app.py
```

The app will open in your browser at `http://localhost:8501`.

---

## 📖 How to Use

1. **Upload PDFs** — Use the sidebar file uploader to select one or more PDF files
2. **Process Documents** — Click the **"📥 Process Docs"** button to extract text, create chunks, and build the search index
3. **Ask Questions** — Type your question in the chat input at the bottom
4. **View Sources** — Expand the "Retrieved Sources" section to see which document chunks were used
5. **Experiment** — Adjust chunk size, overlap, and top-k in the sidebar settings

---

## ⚙️ Configuration Options

| Setting | Default | Range | Description |
|---------|---------|-------|-------------|
| Chunk Size | 300 tokens | 100–800 | Size of each document chunk |
| Chunk Overlap | 50 tokens | 0–200 | Overlap between consecutive chunks |
| Top-K | 5 | 1–10 | Number of chunks to retrieve per query |
| Stream Responses | On | On/Off | Show tokens as they are generated |
| Show Sources | On | On/Off | Display retrieved document chunks |

---

## 📊 Evaluation Features

The **Evaluation** tab provides:

- **Query Logging** — All queries, responses, and metrics are logged to `logs/query_log.jsonl`
- **Response Time Tracking** — Bar chart of response times across queries
- **Chunk Size Comparison** — Compare average response times across different chunk size settings
- **Precision\@K Calculator** — Manually mark relevant sources to calculate retrieval precision
- **Full Query History** — Expandable log of all past queries with full details

---

## 🧠 Technical Details

### Embedding Model

**all-MiniLM-L6-v2** (SentenceTransformers)
- Dimension: 384
- Speed: ~14,000 sentences/sec on CPU
- Size: ~80MB
- No API needed — runs locally

### LLM

**Llama 3 8B Instruct** (via Ollama)
- Parameters: 8 billion
- Quantized for Apple Silicon (Metal GPU acceleration)
- Context window: 8,192 tokens
- No API key required

### Vector Database

**FAISS** (Facebook AI Similarity Search)
- IndexFlatIP (Inner Product = Cosine Similarity with normalized vectors)
- Exact search (no approximation)
- In-memory with disk persistence

### Document Processing

**PyMuPDF (fitz)**
- Fast PDF text extraction
- Preserves reading order
- Handles multi-page documents

---

## 🔧 Troubleshooting

### "Cannot connect to Ollama"
```bash
# Ensure Ollama is running
ollama serve

# Verify it's accessible
curl http://localhost:11434/api/tags
```

### "Model not found"
```bash
# Pull the required model
ollama pull llama3

# Verify it's available
ollama list
```

### Slow responses
- Ensure no other heavy processes are running
- Try a smaller chunk size (200 tokens) for faster retrieval
- Reduce top-k to 3 for shorter context
- Close other applications to free memory

### Memory issues
- Process fewer documents at a time
- Use a smaller chunk size to reduce index size
- The system is optimized for MacBook Air — keep documents under 100 pages total

---

## 📝 License

This project is open-source and uses only open-source dependencies. No paid APIs or cloud services are required.

---

## 🙏 Acknowledgments

- [Ollama](https://ollama.com) — Local LLM runtime
- [Meta Llama 3](https://llama.meta.com) — Open-source language model
- [SentenceTransformers](https://www.sbert.net) — Embedding models
- [FAISS](https://github.com/facebookresearch/faiss) — Vector similarity search
- [Streamlit](https://streamlit.io) — Web application framework
- [PyMuPDF](https://pymupdf.readthedocs.io) — PDF processing
