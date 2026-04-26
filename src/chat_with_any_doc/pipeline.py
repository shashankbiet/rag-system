from dataclasses import dataclass
from typing import Any

from langchain_core.prompts import ChatPromptTemplate

from genai import (
    build_faiss_vectorstore,
    get_chat_model,
    get_embedding_model,
    get_loader,
    get_vectorstore_retriever,
    split_documents,
)


def build_document_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant answering questions from a document."),
            ("human", "Context:\n{context}\n\nQuestion:\n{question}"),
        ]
    )


@dataclass(slots=True)
class DocumentRagPipeline:
    retriever: Any
    chain: Any

    def answer(self, question: str) -> str:
        docs = self.retriever.invoke(question)
        context = "\n\n".join(document.page_content for document in docs)
        response = self.chain.invoke({"context": context, "question": question})
        return response.content


def create_doc_rag(
    file_path: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    retrieval_k: int = 3,
) -> DocumentRagPipeline:
    loader = get_loader(file_path)
    documents = loader.load(file_path)
    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    embedding_model = get_embedding_model()
    vector_store = build_faiss_vectorstore(chunks, embedding_model)
    retriever = get_vectorstore_retriever(vector_store, k=retrieval_k)

    prompt = build_document_prompt()
    chat_model = get_chat_model()
    chain = prompt | chat_model
    return DocumentRagPipeline(retriever=retriever, chain=chain)
