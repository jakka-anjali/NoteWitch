import streamlit as st
import pymupdf as fitz  # PyMuPDF for robust PDF text extraction
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import requests
import json
import time

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="NoteWitch - Offline RAG PDF QA",
    page_icon="🧙‍♀️",
    layout="wide"
)

# --- CACHED RESOURCES ---
@st.cache_resource(show_spinner=False)
def load_embedding_model():
    """Loads and caches the Sentence Transformer embedding model."""
    return SentenceTransformer("all-MiniLM-L6-v2")

def check_ollama_status(base_url="http://localhost:11434"):
    """Checks if the local Ollama server is running and lists available models."""
    try:
        resp = requests.get(f"{base_url}/api/tags", timeout=3)
        if resp.status_code == 200:
            models = [m.get("name") for m in resp.json().get("models", [])]
            return True, models
    except Exception:
        pass
    return False, []

# --- DOCUMENT PROCESSING FUNCTIONS ---
def extract_text_from_pdf(pdf_bytes):
    """Extracts raw text from an uploaded PDF stream page by page."""
    doc = fitz.open(stream=pdf_bytes, filetype="pdf")
    pages_text = []
    for page_num in range(len(doc)):
        page = doc[page_num]
        text = page.get_text("text")
        if text.strip():
            pages_text.append({"page": page_num + 1, "text": text.strip()})
    doc.close()
    return pages_text

def chunk_document_pages(pages_data, chunk_size=600, chunk_overlap=120):
    """
    Splits document text into overlapping chunks while tracking source page numbers.
    Preserves paragraph and sentence boundaries for higher retrieval quality.
    """
    chunks = []
    
    for page_info in pages_data:
        page_num = page_info["page"]
        raw_text = page_info["text"]
        
        # Split on natural paragraph or line breaks
        paragraphs = [p.strip() for p in raw_text.split("\n") if p.strip()]
        
        current_chunk = ""
        for p in paragraphs:
            if len(current_chunk) + len(p) + 1 <= chunk_size:
                current_chunk = (current_chunk + " " + p).strip()
            else:
                if current_chunk:
                    chunks.append({"page": page_num, "content": current_chunk})
                    # Overlap slice
                    overlap = current_chunk[-chunk_overlap:] if len(current_chunk) > chunk_overlap else current_chunk
                    current_chunk = (overlap + " " + p).strip()
                else:
                    # Single paragraph exceeds chunk_size, split forcefully
                    chunks.append({"page": page_num, "content": p[:chunk_size]})
                    current_chunk = p[chunk_size:]
                    
        if current_chunk:
            chunks.append({"page": page_num, "content": current_chunk})
            
    return chunks

def build_faiss_index(chunks, embed_model):
    """Computes normalized embeddings and builds an inner-product (cosine similarity) FAISS index."""
    texts = [c["content"] for c in chunks]
    embeddings = embed_model.encode(texts, convert_to_numpy=True, show_progress_bar=False)
    
    # L2 normalize embeddings for cosine similarity with IndexFlatIP
    faiss.normalize_L2(embeddings)
    
    dim = embeddings.shape[1]
    index = faiss.IndexFlatIP(dim)
    index.add(embeddings)
    
    return index, embeddings

def retrieve_top_k(query, embed_model, index, chunks, k=4):
    """Retrieves top-k most semantically relevant chunks for a given query."""
    q_vec = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_vec)
    
    scores, indices = index.search(q_vec, min(k, len(chunks)))
    
    results = []
    for score, idx in zip(scores[0], indices[0]):
        if idx != -1:
            results.append({
                "score": float(score),
                "page": chunks[idx]["page"],
                "content": chunks[idx]["content"]
            })
    return results

def stream_ollama_response(prompt, model_name="mistral", base_url="http://localhost:11434"):
    """Streams generated response tokens from local Ollama instance."""
    payload = {
        "model": model_name,
        "prompt": prompt,
        "stream": True,
        "options": {
            "temperature": 0.2,
            "top_p": 0.9
        }
    }
    
    try:
        with requests.post(f"{base_url}/api/generate", json=payload, stream=True, timeout=120) as r:
            r.raise_for_status()
            for line in r.iter_lines():
                if line:
                    data = json.loads(line.decode("utf-8"))
                    yield data.get("response", "")
                    if data.get("done", False):
                        break
    except requests.exceptions.ConnectionError:
        yield "\n\n⚠️ **Error:** Could not connect to Ollama. Please ensure the Ollama service is running (`ollama serve`)."
    except Exception as e:
        yield f"\n\n⚠️ **Error:** {str(e)}"

# --- UI SIDEBAR ---
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/wizard.png", width=64)
    st.title("NoteWitch")
    st.caption("Offline RAG PDF Question Answering System")
    
    st.markdown("---")
    st.subheader("⚙️ System Status")
    
    ollama_online, installed_models = check_ollama_status()
    if ollama_online:
        st.success("🟢 Ollama Server: Connected")
        available_mistral = [m for m in installed_models if "mistral" in m.lower()]
        selected_model = st.selectbox(
            "Selected LLM",
            options=installed_models,
            index=installed_models.index(available_mistral[0]) if available_mistral else 0
        )
    else:
        st.error("🔴 Ollama Server: Offline")
        st.info("Start Ollama in your terminal: `ollama serve`")
        selected_model = "mistral"
        
    st.markdown("---")
    st.subheader("📚 Tech Stack")
    st.markdown(
        """
        - **LLM:** Mistral (via Ollama)
        - **Embeddings:** Sentence Transformers (`all-MiniLM-L6-v2`)
        - **Vector DB:** FAISS (Cosine Similarity)
        - **PDF Engine:** PyMuPDF
        - **Deep Learning Backend:** PyTorch
        - **Frontend:** Streamlit
        """
    )
    
    if st.button("🔄 Clear Active Document"):
        st.session_state.clear()
        st.rerun()

# --- MAIN PAGE ---
st.title("🧙‍♀️ NoteWitch — Chat with Your PDF")
st.markdown(
    "Ask detailed questions about your document. Responses are synthesized by a local **Mistral** LLM "
    "grounded strictly on semantic context retrieved via **FAISS** and **Sentence Transformers**."
)

# File Uploader
uploaded_file = st.file_uploader(
    "📄 Upload a PDF document",
    type=["pdf"],
    help="Select a text-based PDF file to begin."
)

# Initialize Session State
if "doc_id" not in st.session_state:
    st.session_state.doc_id = None
    st.session_state.chunks = []
    st.session_state.index = None
    st.session_state.num_pages = 0
    st.session_state.chat_history = []

# Process PDF if uploaded and not already indexed
if uploaded_file is not None:
    file_id = f"{uploaded_file.name}_{uploaded_file.size}"
    
    if st.session_state.doc_id != file_id:
        with st.status("🔮 Analyzing and indexing PDF...", expanded=True) as status:
            st.write("1. Extracting text with PyMuPDF...")
            pages_data = extract_text_from_pdf(uploaded_file.read())
            
            if not pages_data:
                status.update(label="❌ Failed: No extractable text found.", state="error")
                st.error("This PDF does not contain extractable text. Please upload a standard text PDF (not scanned images).")
                st.stop()
                
            st.write(f"2. Extracted text from {len(pages_data)} pages. Chunking text...")
            chunks = chunk_document_pages(pages_data, chunk_size=600, chunk_overlap=120)
            
            st.write(f"3. Generating dense embeddings for {len(chunks)} chunks using Sentence Transformers...")
            embed_model = load_embedding_model()
            index, embeddings = build_faiss_index(chunks, embed_model)
            
            # Store in session state
            st.session_state.doc_id = file_id
            st.session_state.filename = uploaded_file.name
            st.session_state.chunks = chunks
            st.session_state.index = index
            st.session_state.num_pages = len(pages_data)
            st.session_state.chat_history = []
            
            status.update(label=f"✅ Indexing complete: {len(chunks)} chunks indexed across {len(pages_data)} pages!", state="complete")

# Display Document Info Metrics
if st.session_state.doc_id:
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("📄 Active Document", st.session_state.filename)
    col2.metric("📑 Total Pages", st.session_state.num_pages)
    col3.metric("🧩 Indexed Chunks", len(st.session_state.chunks))
    col4.metric("📐 Vector Dimension", 384)
    
    st.markdown("---")
    
    # Question Input
    query = st.chat_input("🔎 Ask a question about this document...")
    
    # Display prior conversation turns
    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            if "sources" in message and message["sources"]:
                with st.expander("🔍 View Retrieved Context Chunks"):
                    for i, src in enumerate(message["sources"], 1):
                        st.markdown(f"**Chunk {i} (Page {src['page']}, Similarity Score: {src['score']:.4f})**")
                        st.caption(src["content"])
    
    if query:
        # Display user query immediately
        st.session_state.chat_history.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
            
        # Retrieve context chunks
        embed_model = load_embedding_model()
        retrieved_sources = retrieve_top_k(query, embed_model, st.session_state.index, st.session_state.chunks, k=4)
        
        # Build grounded context string
        context_text = "\n\n---\n\n".join(
            [f"[Source: Page {s['page']}]:\n{s['content']}" for s in retrieved_sources]
        )
        
        # Prepare System Prompt
        rag_prompt = f"""You are NoteWitch, an expert offline AI assistant for document question answering.
Your task is to provide an accurate, clear, and comprehensive answer to the user's question using ONLY the provided document context below.
Do not fabricate information. If the answer cannot be determined from the context, state honestly that the document does not contain this information.

Context:
{context_text}

User Question:
{query}

Detailed Answer:"""

        # Stream response
        with st.chat_message("assistant"):
            response_container = st.empty()
            full_response = ""
            
            # Use generator streaming
            generator = stream_ollama_response(rag_prompt, model_name=selected_model)
            for token in generator:
                full_response += token
                response_container.markdown(full_response + "▌")
            response_container.markdown(full_response)
            
            # Show sources in expander
            if retrieved_sources:
                with st.expander("🔍 View Retrieved Context Chunks"):
                    for i, src in enumerate(retrieved_sources, 1):
                        st.markdown(f"**Chunk {i} (Page {src['page']}, Similarity Score: {src['score']:.4f})**")
                        st.caption(src["content"])
                        
        # Save assistant turn to chat history
        st.session_state.chat_history.append({
            "role": "assistant",
            "content": full_response,
            "sources": retrieved_sources
        })
else:
    st.info("👆 Please upload a PDF file above to start asking questions.")
