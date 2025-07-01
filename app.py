import streamlit as st
import fitz  # PyMuPDF
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
import os

st.title("🧙‍♀️ NoteWitch - Chat with Your PDF")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    pdf_reader = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in pdf_reader:
        text += page.get_text()

    splitter = CharacterTextSplitter(separator="\n", chunk_size=300, chunk_overlap=50)
    texts = splitter.split_text(text)
    texts = texts[:10]  # limit to avoid overload

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    docsearch = FAISS.from_texts(texts, embeddings)

    query = st.text_input("Ask a question about the PDF:")
    if query:
        docs = docsearch.similarity_search(query)
        st.subheader("Top Matching Sections:")
        for i, doc in enumerate(docs):
            st.markdown(f"**Match {i+1}:** {doc.page_content}")
