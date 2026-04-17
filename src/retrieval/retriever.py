"""
Step 6+7 — Dense retrieval + Cross-encoder reranking.

Pipeline:
  query
    │
    ▼
  ChromaDB cosine similarity  (top_k=10, dense/semantic)
    │
    ▼
  bge-reranker-v2-m3          (cross-encoder, multilingual)
    │
    ▼
  top-4 most relevant chunks  → passed to LLM
"""

from pathlib import Path
from typing import Any

from langchain_core.documents import Document
from langchain_chroma import Chroma
from sentence_transformers import CrossEncoder
from rich.console import Console

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))
import config

console = Console()


class RerankRetriever:
    """
    Dense retriever backed by ChromaDB + cross-encoder reranker.

    Usage:
        retriever = RerankRetriever(vectorstore)
        docs = retriever.retrieve("Chi era Micol Finzi-Contini?")
    """

    def __init__(self, vectorstore: Chroma):
        self.vectorstore = vectorstore

        console.print(
            f"[bold cyan]Loading reranker:[/bold cyan] {config.RERANKER_MODEL}"
        )
        self.reranker = CrossEncoder(
            config.RERANKER_MODEL,
            max_length=512,
            device="cpu",          # change to "mps" on Apple Silicon if desired
        )
        console.print("[green]Reranker ready.[/green]")

    # ── Public API ────────────────────────────────────────────────────────────

    def retrieve(self, query: str) -> list[Document]:
        """
        Full pipeline: dense retrieval → rerank → return top docs.
        """
        # 1. Dense retrieval — E5Embeddings.embed_query() adds "query: " prefix
        candidates: list[Document] = self.vectorstore.similarity_search(
            query, k=config.TOP_K_RETRIEVE
        )

        if not candidates:
            return []

        # 2. Cross-encoder reranking
        pairs = [[query, doc.page_content] for doc in candidates]
        scores: list[float] = self.reranker.predict(pairs).tolist()

        # Attach score to metadata and sort descending
        for doc, score in zip(candidates, scores):
            doc.metadata["rerank_score"] = round(score, 4)

        ranked = sorted(candidates, key=lambda d: d.metadata["rerank_score"],
                        reverse=True)
        top = ranked[: config.TOP_K_RERANK]

        self._log_retrieval(query, top)
        return top

    # ── Helpers ───────────────────────────────────────────────────────────────

    def _log_retrieval(self, query: str, docs: list[Document]) -> None:
        console.print(f"\n[dim]Query:[/dim] {query[:80]}")
        for i, doc in enumerate(docs, 1):
            meta = doc.metadata
            chapter = meta.get("chapter", "?")
            part    = meta.get("part")
            label   = chapter + (f" §{part}" if part else "")
            console.print(
                f"  [cyan]{i}.[/cyan] "
                f"[yellow]{label[:45]}[/yellow]  "
                f"score=[bold]{meta.get('rerank_score',0):.3f}[/bold]  "
                f"chars={meta.get('total_chars','?')}"
            )
