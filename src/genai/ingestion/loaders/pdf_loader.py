from langchain_community.document_loaders import PyPDFLoader
from langchain_core.documents import Document


class PDFLoader:
    def load(self, path: str) -> list[Document]:
        loader = PyPDFLoader(path)
        documents = loader.load()
        return documents
