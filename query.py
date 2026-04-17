"""
query.py — Interactive RAG chat session.

Run:
    python query.py

Commands during the session:
    /reset   — clear conversation history
    /quit    — exit
    /chunks  — show how many docs are in the store
"""

from pathlib import Path
from rich.console import Console
from rich.rule import Rule
from rich.markdown import Markdown

import config
from src.vectorstore.indexer  import load_vectorstore
from src.retrieval.retriever  import RerankRetriever
from src.generation.chain     import RAGChain

console = Console()


def main() -> None:
    console.print(Rule("[bold]Il Giardino dei Finzi-Contini — RAG Chat[/bold]"))

    # Load existing vector store
    vectorstore = load_vectorstore()
    retriever   = RerankRetriever(vectorstore)
    chain       = RAGChain()

    console.print(
        "\n[bold]Pronto![/bold]  Fai domande sul romanzo in italiano o inglese.\n"
        "Comandi: [cyan]/reset[/cyan] [cyan]/quit[/cyan] [cyan]/chunks[/cyan]\n"
    )

    while True:
        try:
            question = console.input("[bold cyan]Tu:[/bold cyan] ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not question:
            continue

        if question.lower() in ("/quit", "/exit", "quit", "exit"):
            break

        if question.lower() == "/reset":
            chain.reset()
            continue

        if question.lower() == "/chunks":
            count = vectorstore._collection.count() if hasattr(vectorstore, "_collection") else len(vectorstore.get()["ids"])
            console.print(f"[dim]Chunks in ChromaDB: {count}[/dim]")
            continue

        # Stage 1: rewrite query for better retrieval
        retrieval_query = chain.rewrite_query(question)
        # Stage 2: retrieve → rerank → generate (only fresh context, no old answers)
        docs   = retriever.retrieve(retrieval_query)
        answer = chain.answer(question, docs, retrieval_query=retrieval_query)

        console.print()
        console.print(Rule("[dim]Risposta[/dim]", style="dim"))
        console.print(Markdown(answer))
        console.print()

    console.print("[dim]Arrivederci![/dim]")


if __name__ == "__main__":
    main()
