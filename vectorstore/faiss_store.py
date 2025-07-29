from langchain.vectorstores import FAISS
from langchain.embeddings import HuggingFaceEmbeddings

def create_vector_store(chunks):
    embedder = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = FAISS.from_documents(chunks, embedder)
    return vectorstore
