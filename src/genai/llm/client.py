from langchain_core.language_models.chat_models import BaseChatModel
from llm_factory import LLMFactory, Provider

from config.settings import settings


def get_chat_model() -> BaseChatModel:
    chat_model = LLMFactory.create_chat_model(Provider.LMSTUDIO, model=settings.chat_model, temperature=0.7)
    return chat_model
