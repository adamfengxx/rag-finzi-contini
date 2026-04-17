"""
Step 4+5 — Embed documents and persist to ChromaDB.

Embedding model: intfloat/multilingual-e5-large
  - Top-tier Italian/multilingual performance
  - Free, runs locally via sentence-transformers
  - Requires E5 instruction prefixes ("passage: " for indexing,
    "query: " at query time)

ChromaDB is used in persistent mode so the index survives between runs.
"""

from pathlib import Path
from typing import List

from langchain_core.documents import Document
from langchain_core.embeddings import Embeddings
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from rich.console import Console

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))
import config

console = Console()


class E5Embeddings(Embeddings):
    """
    Wraps HuggingFaceEmbeddings and prepends the E5 instruction prefix:
      - embed_documents → "passage: " + text
      - embed_query     → "query: "   + text

    Newer langchain-huggingface dropped query_instruction/embed_instruction,
    so we handle prefixing here manually.
    """

    def __init__(self):
        self._model = HuggingFaceEmbeddings(
            model_name=config.EMBEDDING_MODEL,
            encode_kwargs={"normalize_embeddings": True},
        )

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        prefixed = [config.E5_PASSAGE_PREFIX + t for t in texts]
        return self._model.embed_documents(prefixed)

    def embed_query(self, text: str) -> List[float]:
        return self._model.embed_query(config.E5_QUERY_PREFIX + text)


def build_vectorstore(docs: list[Document]) -> Chroma:
    """
    Embed *docs* and persist the ChromaDB collection.
    Always drops the existing collection first — re-runs are always clean.
    """
    console.print(
        f"[bold cyan]Building vector store[/bold cyan]  "
        f"model=[italic]{config.EMBEDDING_MODEL}[/italic]  "
        f"docs={len(docs)}"
    )

    # Drop existing collection before rebuild to prevent duplicate data
    import chromadb
    client   = chromadb.PersistentClient(path=str(config.CHROMA_DIR))
    existing = [c.name for c in client.list_collections()]
    if config.CHROMA_COLLECTION in existing:
        client.delete_collection(config.CHROMA_COLLECTION)
        console.print(
            f"[yellow]Dropped existing collection '{config.CHROMA_COLLECTION}' — rebuilding clean.[/yellow]"
        )

    embed_fn = E5Embeddings()

    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embed_fn,
        collection_name=config.CHROMA_COLLECTION,
        persist_directory=str(config.CHROMA_DIR),
        collection_metadata={"hnsw:space": "cosine"},
    )

    console.print(
        f"[green]Vector store ready.[/green]  "
        f"Collection '{config.CHROMA_COLLECTION}' @ {config.CHROMA_DIR}"
    )
    return vectorstore


def load_vectorstore() -> Chroma:
    """Load an existing ChromaDB collection (no re-embedding)."""
    embed_fn = E5Embeddings()
    vectorstore = Chroma(
        collection_name=config.CHROMA_COLLECTION,
        embedding_function=embed_fn,
        persist_directory=str(config.CHROMA_DIR),
    )
    count = vectorstore._collection.count() if hasattr(vectorstore, "_collection") else len(vectorstore.get()["ids"])
    console.print(
        f"[green]Loaded vector store:[/green] "
        f"'{config.CHROMA_COLLECTION}'  ({count} vectors)"
    )
    return vectorstore
