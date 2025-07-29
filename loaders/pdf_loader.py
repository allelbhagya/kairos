import os
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

def load_and_chunk_pdfs(data_dir="./data"):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    all_chunks = []

    for file in os.listdir(data_dir):
        if file.endswith(".pdf"):
            loader = PyPDFLoader(os.path.join(data_dir, file))
            docs = loader.load()
            chunks = splitter.split_documents(docs)
            all_chunks.extend(chunks)

    return all_chunks
