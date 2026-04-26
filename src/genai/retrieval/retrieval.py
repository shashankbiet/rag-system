from langchain_core.retrievers import BaseRetriever
from langchain_core.vectorstores import VectorStore


def get_vectorstore_retriever(
    vectorstore: VectorStore,
    k: int = 3,
) -> BaseRetriever:
    return vectorstore.as_retriever(search_kwargs={"k": k})
