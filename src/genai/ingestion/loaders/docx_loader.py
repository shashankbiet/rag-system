import logging

from langchain_community.document_loaders import Docx2txtLoader
from langchain_core.documents import Document

logger = logging.getLogger(__name__)


class DocxLoader:
    def load(self, path: str) -> list[Document]:
        loader = Docx2txtLoader(path)
        documents = loader.load()
        logger.info(f"DOCX loaded successfully. Document count: {len(documents)}")
        return documents
