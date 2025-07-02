import streamlit as st
import fitz  # PyMuPDF
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import Ollama
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA

st.title("🧙‍♀️ NoteWitch - Chat with Your PDF")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

if uploaded_file:
    pdf_reader = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in pdf_reader:
        text += page.get_text()

    splitter = CharacterTextSplitter(separator="\n", chunk_size=300, chunk_overlap=50)
    texts = splitter.split_text(text)

    # ✅ Online embedding model
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")
    docsearch = FAISS.from_texts(texts, embeddings)

    query = st.text_input("Ask a question about the PDF:")

    if query:
        llm = Ollama(model="mistral")  # still use Ollama for generation
        qa_chain = RetrievalQA.from_chain_type(llm=llm, retriever=docsearch.as_retriever())
        response = qa_chain.run(query)

        st.subheader("🧠 Answer:")
        st.write(response)
