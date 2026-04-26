# RAG System

Reference implementations for Retrieval-Augmented Generation workflows.

The repository is structured as a foundation for multiple RAG systems. The goal is to keep reusable RAG infrastructure separate from app-specific workflows so new examples can be added without duplicating ingestion, embedding, retrieval, or LLM code.

## Projects

### Chat With Any Document

Ask questions over local documents using ingestion, chunking, embeddings, FAISS retrieval, and a local chat model.

Package:

```text
src/chat_with_any_doc/
```

CLI entrypoint:

```text
src/app/cli/chat_with_any_doc.py
```

Run:

```bash
uv run chat-with-any-doc
```

Default input:

```text
data/sample.pdf
```

Supported document types:

- PDF
- DOCX
- TXT

Workflow:

```text
Document
  ↓
Loader
  ↓
Text splitter
  ↓
Embedding model
  ↓
FAISS vector store
  ↓
Retriever
  ↓
Prompt + retrieved context
  ↓
Chat model
  ↓
Answer
```

Notes:

- Uses an in-memory FAISS index.
- Rebuilds the vector index on every run.
- Does not persist embeddings or retrieved context.


## Repository Layout

```text
rag-system/
├── data/
├── src/
│   ├── app/
│   │   └── cli/
│   ├── config/
│   ├── genai/
│   │   ├── embedding/
│   │   ├── ingestion/
│   │   ├── llm/
│   │   └── retrieval/
│   └── <rag_project>/
├── pyproject.toml
├── requirements.txt
└── uv.lock
```

## Core Design

The code is split into three layers:

`genai`

Reusable RAG primitives. This package owns shared capabilities such as loading, splitting, embedding model creation, vector-store creation, retriever creation, and chat model creation.

`<rag_project>`

A project-specific workflow package. Each workflow composes the reusable `genai` pieces into a concrete RAG system.

`app`

Application entrypoints. CLI, API, or UI entrypoints should stay thin and delegate workflow logic to the project package.

## Tech Stack

- Python 3.13+
- LangChain
- FAISS
- LM Studio-compatible local models via `llm-factory`
- Pydantic Settings
- uv
- Ruff

## Setup

Create and activate a virtual environment:

```bash
uv venv
source .venv/bin/activate
```

Install dependencies:

```bash
uv sync
```

Run lint checks:

```bash
uv run ruff check src
```

## Local Model Setup

The default settings expect LM Studio-compatible local models:

```python
chat_model = "gemma-3-1b"
embedding_model = "nomic-embed-text-v1.5"
```

Start the LM Studio local server with an OpenAI-compatible endpoint:

```text
http://localhost:1234/v1
```

Model names can be changed in:

```text
src/config/settings.py
```
