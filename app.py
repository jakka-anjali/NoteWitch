import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

st.title("🧙‍♀️ NoteWitch - Chat with Your PDF")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    pdf_reader = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in pdf_reader:
        text += page.get_text()

    # Split text into chunks
    chunks = text.split("\n\n")
    chunks = [chunk.strip() for chunk in chunks if chunk.strip()]

    # Embed chunks
    model = SentenceTransformer("all-MiniLM-L6-v2")
    embeddings = model.encode(chunks)

    # Create FAISS index
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    # Accept query
    query = st.text_input("Ask a question about the PDF:")
    if query:
        q_embedding = model.encode([query])
        D, I = index.search(q_embedding, k=3)

        context = "\n\n".join([chunks[i] for i in I[0]])

        # Load QA model
        qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

        answer = qa(question=query, context=context)

        st.subheader("🧠 Answer:")
        st.write(answer["answer"])
