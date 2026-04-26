import argparse
import logging
import os

os.environ["TOKENIZERS_PARALLELISM"] = "false"

from chat_with_youtube_video.pipeline import create_youtube_rag
from config.logging_config import setup_logging

logger = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description="Chat with a YouTube video transcript.")
    parser.add_argument("video", help="YouTube video ID or URL")
    args = parser.parse_args()

    setup_logging()

    rag = create_youtube_rag(video=args.video)

    while True:
        question = input("\nAsk a question (or 'exit'): ")
        if question.lower() == "exit":
            break

        print("\nAnswer:\n", rag.answer(question))
