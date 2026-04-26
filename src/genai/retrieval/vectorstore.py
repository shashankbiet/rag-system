from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_core.vectorstores import VectorStore


def build_faiss_vectorstore(documents: list[Document], embeddings: Embeddings) -> VectorStore:
    vectorstore = FAISS.from_documents(documents, embeddings)
    return vectorstore
