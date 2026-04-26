from langchain_core.embeddings import Embeddings
from llm_factory import LLMFactory, Provider

from config.settings import settings


def get_embedding_model() -> Embeddings:
    embeddings = LLMFactory.create_embedding_model(Provider.LMSTUDIO, model=settings.embedding_model)
    return embeddings
