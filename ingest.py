"""
ingest.py — Run the full ingestion pipeline once.

Steps:
  1. EPUB → Markdown
  2. Preprocess / clean
  3. Chunk with metadata
  4. Embed & persist to ChromaDB

Run:
    python ingest.py
    python ingest.py --skip-convert   # if .md already exists
"""

import argparse
from pathlib import Path

from rich.console import Console
from rich.rule import Rule

import config
from src.ingestion.epub_converter import convert_epub_to_markdown
from src.ingestion.preprocessor   import preprocess_file
from src.chunking.chunker          import build_documents, save_chunks_json
from src.vectorstore.indexer       import build_vectorstore

console = Console()


def main(skip_convert: bool = False) -> None:
    console.print(Rule("[bold]Il Giardino dei Finzi-Contini — RAG Ingestion[/bold]"))

    # ── 1. EPUB → Markdown ────────────────────────────────────────────────────
    if skip_convert and config.MD_FILE.exists():
        console.print(f"[yellow]Skipping conversion (--skip-convert)[/yellow]  "
                      f"Using {config.MD_FILE}")
    else:
        console.print(Rule("Step 1 · EPUB → Markdown"))
        convert_epub_to_markdown(config.EPUB_FILE, config.MD_FILE)

    # ── 2. Preprocess ─────────────────────────────────────────────────────────
    console.print(Rule("Step 2 · Preprocessing"))
    preprocess_file(config.MD_FILE)          # in-place cleaning

    # ── 3. Chunk ──────────────────────────────────────────────────────────────
    console.print(Rule("Step 3 · Chunking"))
    docs = build_documents(config.MD_FILE)
    save_chunks_json(docs, config.CHUNKS_DIR / "chunks.json")

    # ── 4. Embed + index ──────────────────────────────────────────────────────
    console.print(Rule("Step 4 · Embedding & Indexing"))
    build_vectorstore(docs)

    console.print(Rule("[bold green]Ingestion complete[/bold green]"))
    console.print(
        f"[green]✓[/green] {len(docs)} chunks indexed in ChromaDB @ {config.CHROMA_DIR}\n"
        f"[green]✓[/green] Run [bold]python query.py[/bold] to start querying."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest EPUB into ChromaDB")
    parser.add_argument("--skip-convert", action="store_true",
                        help="Skip EPUB→Markdown step if .md already exists")
    args = parser.parse_args()
    main(skip_convert=args.skip_convert)
