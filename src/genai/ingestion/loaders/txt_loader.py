from langchain_core.documents import Document


class TxtLoader:
    def load(self, path: str) -> list[Document]:
        with open(path, encoding="utf-8") as f:
            text = f.read()
        return [Document(page_content=text)]
