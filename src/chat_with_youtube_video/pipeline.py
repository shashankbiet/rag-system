from dataclasses import dataclass
from typing import Any
from urllib.parse import parse_qs, urlparse

from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from youtube_transcript_api import YouTubeTranscriptApi, YouTubeTranscriptApiException

from genai import (
    build_faiss_vectorstore,
    get_chat_model,
    get_embedding_model,
    get_vectorstore_retriever,
    split_documents,
)


def extract_video_id(video: str) -> str:
    parsed_url = urlparse(video)
    if parsed_url.netloc in {"youtu.be", "www.youtu.be"}:
        return parsed_url.path.lstrip("/")

    if "youtube.com" in parsed_url.netloc:
        path_parts = [part for part in parsed_url.path.split("/") if part]
        if len(path_parts) > 1 and path_parts[0] in {"embed", "live", "shorts"}:
            return path_parts[1]

        video_id = parse_qs(parsed_url.query).get("v", [""])[0]
        if video_id:
            return video_id

    return video


def build_youtube_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages(
        [
            ("system", "You are a helpful assistant answering questions from a YouTube video transcript."),
            ("human", "Transcript context:\n{context}\n\nQuestion:\n{question}"),
        ]
    )


def load_youtube_transcript(video: str, languages: tuple[str, ...] = ("en",)) -> list[Document]:
    video_id = extract_video_id(video)

    try:
        transcript = YouTubeTranscriptApi().fetch(video_id, languages=languages)
    except YouTubeTranscriptApiException as exc:
        raise ValueError(f"Could not load transcript for video: {video}") from exc

    transcript_text = " ".join(snippet.text for snippet in transcript)
    metadata = {
        "language": transcript.language,
        "language_code": transcript.language_code,
        "source": "youtube",
        "video_id": video_id,
    }
    return [Document(page_content=transcript_text, metadata=metadata)]


@dataclass(slots=True)
class YouTubeRagPipeline:
    retriever: Any
    chain: Any

    def answer(self, question: str) -> str:
        docs = self.retriever.invoke(question)
        context = "\n\n".join(document.page_content for document in docs)
        response = self.chain.invoke({"context": context, "question": question})
        return response.content


def create_youtube_rag(
    video: str,
    chunk_size: int = 1000,
    chunk_overlap: int = 200,
    retrieval_k: int = 3,
    languages: tuple[str, ...] = ("en",),
) -> YouTubeRagPipeline:
    documents = load_youtube_transcript(video, languages=languages)
    chunks = split_documents(documents, chunk_size=chunk_size, chunk_overlap=chunk_overlap)

    embedding_model = get_embedding_model()
    vector_store = build_faiss_vectorstore(chunks, embedding_model)
    retriever = get_vectorstore_retriever(vector_store, k=retrieval_k)

    prompt = build_youtube_prompt()
    chat_model = get_chat_model()
    chain = prompt | chat_model
    return YouTubeRagPipeline(retriever=retriever, chain=chain)
