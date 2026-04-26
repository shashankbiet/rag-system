import logging
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from chat_with_any_doc.pipeline import create_doc_rag
from config.logging_config import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    setup_logging()

    rag = create_doc_rag(file_path="data/sample.pdf")

    while True:
        question = input("\nAsk a question (or 'exit'): ")
        if question.lower() == "exit":
            break

        print("\nAnswer:\n", rag.answer(question))
