import streamlit as st
import fitz  # PyMuPDF
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain.llms import HuggingFaceHub
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
    # texts = texts[:10]  # Optional limit — can remove later

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    docsearch = FAISS.from_texts(texts, embeddings)

    query = st.text_input("Ask a question about the PDF:")

    if query:
        # Use Hugging Face model from HuggingFaceHub
        llm = HuggingFaceHub(
            repo_id="google/flan-t5-large",
            model_kwargs={"temperature": 0.5, "max_length": 512}
        )

        qa_chain = RetrievalQA.from_chain_type(
            llm=llm,
            chain_type="stuff",
            retriever=docsearch.as_retriever()
        )

        response = qa_chain.run(query)

        st.subheader("🧠 Answer:")
        st.write(response)
