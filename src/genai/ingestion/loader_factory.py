import os

from genai.ingestion.loaders.base import DocumentLoader
from genai.ingestion.loaders.docx_loader import DocxLoader
from genai.ingestion.loaders.pdf_loader import PDFLoader
from genai.ingestion.loaders.txt_loader import TxtLoader


def get_loader(path: str) -> DocumentLoader:
    ext = os.path.splitext(path)[1].lower()
    if ext == ".pdf":
        return PDFLoader()
    elif ext == ".docx":
        return DocxLoader()
    elif ext == ".txt":
        return TxtLoader()
    else:
        raise ValueError(f"Unsupported file type: {ext}")
