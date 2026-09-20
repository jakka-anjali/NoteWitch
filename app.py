import streamlit as st
import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline

st.set_page_config(
    page_title="NoteWitch",
    page_icon="🧙‍♀️",
    layout="centered"
)

st.title("🧙‍♀️ NoteWitch - Chat with Your PDF")
st.write("Upload a PDF and ask questions about its contents.")

uploaded_file = st.file_uploader(
    "📄 Upload your PDF",
    type="pdf"
)

if uploaded_file:
    # Extract text from PDF
    pdf_reader = fitz.open(
        stream=uploaded_file.read(),
        filetype="pdf"
    )

    text = ""

    for page in pdf_reader:
        text += page.get_text()

    pdf_reader.close()

    if not text.strip():
        st.error(
            "This PDF does not contain extractable text. "
            "Please upload a text-based PDF."
        )
        st.stop()

    # Split text into chunks
    chunks = text.split("\n\n")
    chunks = [
        chunk.strip()
        for chunk in chunks
        if chunk.strip()
    ]

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Create embeddings
    embeddings = model.encode(chunks)

    # Create FAISS index
    dim = embeddings.shape[1]

    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings))

    st.success("✅ PDF processed successfully!")

    # Accept query
    query = st.text_input(
        "🔎 Ask a question about the PDF:"
    )

    if query:
        # Embed query
        q_embedding = model.encode([query])

        # Retrieve most relevant chunks
        D, I = index.search(q_embedding, k=min(3, len(chunks)))

        context = "\n\n".join(
            [chunks[i] for i in I[0]]
        )

        # Load Question Answering model
        qa = pipeline(
            task="question-answering",
            model="deepset/roberta-base-squad2",
            framework="pt"
        )

        # Generate answer from retrieved context
        answer = qa(
            question=query,
            context=context
        )

        st.subheader("🧠 Answer:")
        st.write(answer["answer"])

        st.caption(
            f"Confidence: {answer['score']:.2%}"
        )

