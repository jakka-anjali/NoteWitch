import streamlit as st
import fitz  # PyMuPDF
from langchain_community.vectorstores import FAISS
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains.question_answering import load_qa_chain
from langchain.llms import OpenAI
import os

st.title("🧙‍♀️ NoteWitch - Chat with Your PDF")

uploaded_file = st.file_uploader("Upload your PDF", type="pdf")

openai_api_key = st.text_input("Enter your OpenAI API Key", type="password")

if uploaded_file and openai_api_key:
    pdf_reader = fitz.open(stream=uploaded_file.read(), filetype="pdf")
    text = ""
    for page in pdf_reader:
        text += page.get_text()

    splitter = CharacterTextSplitter(separator="\n", chunk_size=1000, chunk_overlap=200)
    texts = splitter.split_text(text)

    os.environ["OPENAI_API_KEY"] = openai_api_key

    embeddings = OpenAIEmbeddings()
    docsearch = FAISS.from_texts(texts, embeddings)

    query = st.text_input("Ask a question about the PDF:")
    if query:
        docs = docsearch.similarity_search(query)
        chain = load_qa_chain(OpenAI(), chain_type="stuff")
        response = chain.run(input_documents=docs, question=query)
        st.write("🧠 Answer:", response)

