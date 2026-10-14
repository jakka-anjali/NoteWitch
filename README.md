# 🧙‍♀️ NoteWitch — Offline RAG PDF Question Answering System

[![Python](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12%20%7C%203.13-blue.svg)](https://www.python.org/)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.30+-FF4B4B.svg)](https://streamlit.io/)
[![Ollama](https://img.shields.io/badge/Ollama-Mistral%207B-black.svg)](https://ollama.com/)
[![FAISS](https://img.shields.io/badge/Vector%20DB-FAISS-008080.svg)](https://github.com/facebookresearch/faiss)
[![SentenceTransformers](https://img.shields.io/badge/Embeddings-Sentence%20Transformers-yellowgreen.svg)](https://www.sbert.net/)
[![PyTorch](https://img.shields.io/badge/Backend-PyTorch-EE4C2C.svg)](https://pytorch.org/)

**NoteWitch** is a local, privacy-first document question-answering application built using **Retrieval-Augmented Generation (RAG)**. It allows users to upload PDF documents and ask questions in natural language, receiving comprehensive, grounded answers synthesized by a local **Mistral** large language model running through **Ollama**.

The entire pipeline—document parsing, semantic chunking, embedding generation, vector indexing, similarity search, and generative response synthesis—runs **100% offline** on your machine with zero external API calls or cloud dependencies.

---

## 🌟 Key Features

- **100% Offline & Private:** No data ever leaves your computer. No OpenAI API keys or internet connection required.
- **Deep Generative QA:** Employs **Mistral 7B** to synthesize detailed, multi-sentence explanations rather than shallow extractive snippet matching.
- **Dense Semantic Retrieval:** Generates 384-dimensional dense vector embeddings with **Sentence Transformers** (`all-MiniLM-L6-v2`) powered by **PyTorch**.
- **High-Performance Vector Index:** Uses **FAISS** (`IndexFlatIP` with L2 normalization) for millisecond cosine-similarity search.
- **Context Preservation Chunking:** Sliding-window text chunker with configurable overlap to prevent context fragmentation at page and paragraph boundaries.
- **Real-Time Token Streaming:** Dynamic token streaming via Streamlit for a fast, responsive user experience.
- **Explainable RAG:** View the exact retrieved source chunks, similarity scores, and page references directly in the UI.
- **In-Memory Caching:** Leverages Streamlit resource caching and session state so uploaded PDFs are parsed and embedded once, enabling instant follow-up questions.

---

## 🏗️ System Architecture

```
                                  ┌────────────────────────┐
                                  │   User uploads PDF     │
                                  └───────────┬────────────┘
                                              │
                                              ▼
                                  ┌────────────────────────┐
                                  │   PyMuPDF (fitz)       │
                                  │  Page Text Extraction  │
                                  └───────────┬────────────┘
                                              │
                                              ▼
                                  ┌────────────────────────┐
                                  │   Overlapping Chunking │
                                  │ (Preserves boundaries) │
                                  └───────────┬────────────┘
                                              │
                                              ▼
                                  ┌────────────────────────┐
                                  │  Sentence Transformers │
                                  │  (all-MiniLM-L6-v2)    │
                                  │  [PyTorch Tensor Core] │
                                  └───────────┬────────────┘
                                              │
                                              ▼
                                  ┌────────────────────────┐
                                  │   FAISS Vector Index   │
                                  │  (Cosine Similarity)   │
                                  └───────────┬────────────┘
                                              │
                     ┌────────────────────────┴────────────────────────┐
                     ▼                                                 ▼
        ┌─────────────────────────┐                       ┌─────────────────────────┐
        │   User asks a question  │                       │  Top-k Context Chunks   │
        └────────────┬────────────┘                       └────────────┬────────────┘
                     │                                                 │
                     └────────────────────────┬────────────────────────┘
                                              ▼
                                  ┌────────────────────────┐
                                  │ Context Augmented      │
                                  │ System Prompt          │
                                  └───────────┬────────────┘
                                              │
                                              ▼
                                  ┌────────────────────────┐
                                  │  Ollama Runtime        │
                                  │  Mistral 7B LLM        │
                                  └───────────┬────────────┘
                                              │
                                              ▼
                                  ┌────────────────────────┐
                                  │   Streamlit Web UI     │
                                  │ (Real-Time Streaming)  │
                                  └────────────────────────┘
```

---

## 🛠️ Tech Stack & Implementation Details

| Technology | Role | Implementation in NoteWitch |
| :--- | :--- | :--- |
| **Python** | Core Language | Application logic, data manipulation, and pipeline orchestration. |
| **Mistral 7B** | Generative LLM | Generates coherent, contextual, and grounded answers from retrieved context. |
| **Ollama** | Local LLM Server | Serves Mistral locally on `http://localhost:11434` with zero cloud fees. |
| **FAISS** | Vector Database | Performs sub-millisecond nearest-neighbor search using cosine similarity. |
| **Sentence Transformers** | Dense Embeddings | Encodes text chunks into dense 384-dimensional semantic vectors (`all-MiniLM-L6-v2`). |
| **PyTorch** | Deep Learning Engine | Underpins the Transformer tensor operations and model execution. |
| **PyMuPDF (`fitz`)** | Document Parser | Fast, lossless PDF text extraction retaining page numbers. |
| **Streamlit** | Frontend UI | Responsive web interface featuring chat history, metric widgets, and token streaming. |

---

## 🚀 Quickstart Guide

### 1. Prerequisites
- **Python:** Python 3.10, 3.11, 3.12, or 3.13 installed.
- **Ollama:** Installed from [ollama.com](https://ollama.com/).
  Pull the Mistral model in your terminal:
  ```bash
  ollama pull mistral
  ```
  Ensure Ollama is running (`ollama serve` or background system tray service).

---

### 2. Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jakka-anjali/NoteWitch.git
   cd NoteWitch
   ```

2. **Create and activate a virtual environment:**
   - **Windows (PowerShell):**
     ```powershell
     python -m venv .venv
     .\.venv\Scripts\Activate.ps1
     ```
   - **macOS / Linux:**
     ```bash
     python3 -m venv .venv
     source .venv/bin/activate
     ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

---

### 3. Running the Application

Launch NoteWitch using Streamlit:
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 💡 How to Use

1. **Upload Document:** Click **"Upload your PDF"** and choose any text-based PDF (lecture notes, research papers, reports, manuals).
2. **Indexing:** NoteWitch automatically extracts text, chunks the pages with sliding overlap, and builds an in-memory FAISS vector index.
3. **Ask Questions:** Type your inquiry in the chat bar.
4. **Inspect Sources:** Expand the **"View Retrieved Context Chunks"** dropdown under each response to see the exact text snippets, similarity scores, and page numbers used to formulate the answer.

---

## 📂 Project Structure

```
NoteWitch/
├── .devcontainer/        # Container configuration for Codespaces / Docker
├── .env.example          # Sample environment variables (optional offline config)
├── .gitignore            # Git ignore rules for venv, cache, and secrets
├── app.py                # Main Streamlit RAG application
├── requirements.txt      # Clean, production-ready Python dependencies
└── README.md             # Project documentation and setup guide
```

---

## 📤 Git & GitHub Deployment Instructions

To push your latest changes to GitHub:

```bash
# 1. Initialize git (if not already initialized in your project folder)
git init

# 2. Stage all files (respecting .gitignore)
git add .

# 3. Commit your changes
git commit -m "feat: complete offline RAG pipeline with Mistral, Ollama, FAISS, and Streamlit streaming"

# 4. Set branch to main
git branch -M main

# 5. Add remote (replace with your GitHub repository URL)
git remote add origin https://github.com/jakka-anjali/NoteWitch.git

# 6. Push to GitHub
git push -u origin main --force
```

---

## 📄 License
This project is open-source and available under the [MIT License](LICENSE).
