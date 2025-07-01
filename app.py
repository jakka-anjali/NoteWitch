import streamlit as st
import fitz  # PyMuPDF
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import CharacterTextSplitter
from langchain.llms import HuggingFaceHub
import os

HUGGINGFACE_API_KEY = st.secrets["HUGGINGFACEHUB_API_TOKEN"]
os.environ["HUGGINGFACEHUB_API_TOKEN"] = HUGGINGFACE_API_KEY

st.title("🧙‍♀️ NoteWitch - Chat with Your PDF")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    pdf_reader = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in pdf_reader:
        text += page.get_text()

    splitter = CharacterTextSplitter(separator="\n", chunk_size=300, chunk_overlap=50)
    texts = splitter.split_text(text)

    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    docsearch = FAISS.from_texts(texts, embeddings)

    query = st.text_input("Ask a question about the PDF:")

    if query:
        docs = docsearch.similarity_search(query, k=3)
        context = "\n\n".join([doc.page_content for doc in docs])

        llm = HuggingFaceHub(
            repo_id="google/flan-t5-base",
            model_kwargs={"temperature": 0.5, "max_length": 512}
        )

        prompt = f"Answer the question based on the context below:\n\nContext:\n{context}\n\nQuestion: {query}\nAnswer:"

        response = llm(prompt)

        st.subheader("🧠 Answer:")
        st.write(response)
