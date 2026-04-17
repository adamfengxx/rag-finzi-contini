"""
Step 3 — Chapter-aware chunking.

Book structure (after epub_converter):
  ## Prologo              ← H2: Part-level heading
  ### Prologo — 1         ← H3: Numbered section (part + section number)
  ### Prologo — 2
  ## Parte prima
  ### Parte prima — 1
  ### Parte prima — 2
  ...

Strategy:
  1. Split on H3 (### …) as the primary unit — these are the actual
     numbered sections (finest granularity Bassani uses).
  2. The H2 before each H3 group gives the Part title; extracted from
     the H3 heading text itself (e.g. "Parte prima — 1" → part="Parte prima").
  3. Within each H3 section, use RecursiveCharacterTextSplitter for
     fixed-size overlapping chunks.
  4. Every chunk carries full metadata:
       book_title, author, language,
       part_title, part_number,
       section_number (within the part),
       chunk_index, chunk_id,
       char_start, char_end, total_chars

Why these chunk sizes?
  - 700 chars ≈ 160-200 Italian tokens — enough for a coherent paragraph.
  - 120-char overlap (~17%) prevents mid-sentence boundary loss.
"""

import re
import json
from dataclasses import dataclass, asdict
from pathlib import Path

from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from rich.console import Console
from rich.table import Table

import sys
sys.path.insert(0, str(Path(__file__).parents[2]))
import config

console = Console()


@dataclass
class ChunkMetadata:
    book_title:  str
    author:      str
    language:    str
    chapter:     str        # H2 title  e.g. "Parte prima", "Prologo"
    part:        int | None # H3 section number within the chapter (1, 2, 3 …)
                            # None for chapters with no numbered sub-sections
                            # (Vita e morte di Micòl, Prologo, Epilogo)
    chunk_index: int        # chunk index within this section
    chunk_id:    str        # globally unique  e.g. "Parte prima_s02_c003"
    char_start:  int
    char_end:    int
    total_chars: int


# ── Parsing helpers ───────────────────────────────────────────────────────────

def _parse_sections(text: str) -> list[tuple[str, int, int, str]]:
    """
    Return list of (part_title, part_number, section_number, body_text).

    We split on ### headings (H3 = numbered sections).
    The H2 context is maintained as 'current_part'.
    Pre-heading text (Montale review etc.) → part_number=0, section_number=0.
    A bare ## heading with no ### children (e.g. a Part title page that has
    no body) is skipped — it only updates current_part.
    """
    # Split keeping both H2 and H3 delimiters
    tokens = re.split(r"(?m)^(#{2,3} .+)$", text)
    # tokens = [pre, heading, body, heading, body, …]

    sections: list[tuple[str, int, int, str]] = []
    current_part   = "Prefazione"
    part_number    = 0
    section_number = 0

    # Pre-heading body (reviews / front matter)
    if tokens[0].strip():
        sections.append(("Prefazione", 0, 0, tokens[0].strip()))

    i = 1
    while i < len(tokens) - 1:
        heading = tokens[i]          # e.g. "## Parte prima" or "### Parte prima — 3"
        body    = tokens[i + 1].strip() if i + 1 < len(tokens) else ""
        i      += 2

        if heading.startswith("## "):
            # Part-level heading — update context, no body expected here
            current_part   = heading[3:].strip()
            part_number   += 1
            section_number = 0
            # If there IS prose directly under a ## (shouldn't happen but
            # handle gracefully)
            if body:
                sections.append((current_part, part_number, 0, body))

        elif heading.startswith("### "):
            # Section heading: "### Parte prima — 3"
            raw = heading[4:].strip()
            # Extract numeric section number from the tail "— N"
            m = re.search(r"—\s*(\d+)\s*$", raw)
            if m:
                section_number = int(m.group(1))
                # part_title is everything before " — N"
                part_from_heading = raw[: m.start()].strip()
                # Use converter-embedded part name (more reliable than tracking H2)
                current_part = part_from_heading
            else:
                section_number += 1

            if body:
                sections.append((current_part, part_number, section_number, body))

    return sections


# ── Public API ────────────────────────────────────────────────────────────────

def build_documents(md_path: Path) -> list[Document]:
    """
    Read the preprocessed Markdown and return a list of LangChain Documents
    with full structural metadata on every chunk.
    """
    text     = md_path.read_text(encoding="utf-8")
    sections = _parse_sections(text)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=config.CHUNK_SIZE,
        chunk_overlap=config.CHUNK_OVERLAP,
        separators=["\n\n", "\n", ".", "!", "?", ";", ",", " "],
        length_function=len,
        add_start_index=True,
    )

    all_docs: list[Document] = []

    for part_title, _, sec_num, sec_text in sections:
        # sec_num == 0 means this section has no numbered sub-part (Prologo etc.)
        part_val = sec_num if sec_num != 0 else None

        chunks = splitter.create_documents(
            texts=[sec_text],
            metadatas=[{}],
        )

        for local_idx, chunk in enumerate(chunks):
            # skip separator artefacts (e.g. lone "---")
            if len(chunk.page_content.strip()) < 20:
                continue
            char_start = chunk.metadata.get("start_index", 0)
            char_end   = char_start + len(chunk.page_content)

            sec_tag  = f"s{sec_num:02d}" if part_val is not None else "s00"
            meta = ChunkMetadata(
                book_title  = config.BOOK_TITLE,
                author      = config.BOOK_AUTHOR,
                language    = config.BOOK_LANG,
                chapter     = part_title,
                part        = part_val,
                chunk_index = local_idx,
                chunk_id    = f"{part_title}_{sec_tag}_c{local_idx:03d}",
                char_start     = char_start,
                char_end       = char_end,
                total_chars    = len(chunk.page_content),
            )

            all_docs.append(Document(
                page_content=chunk.page_content,
                metadata=asdict(meta),
            ))

    # ── Summary table ─────────────────────────────────────────────────────────
    part_counts: dict[str, int] = {}
    for doc in all_docs:
        pt = doc.metadata["chapter"]
        part_counts[pt] = part_counts.get(pt, 0) + 1

    table = Table(title="[bold]Chunking Summary[/bold]", show_header=True)
    table.add_column("Part",   style="cyan")
    table.add_column("Chunks", style="magenta", justify="right")
    for pt, cnt in part_counts.items():
        table.add_row(pt, str(cnt))
    console.print(table)

    avg = sum(len(d.page_content) for d in all_docs) // len(all_docs)
    console.print(
        f"[green]Chunking complete:[/green] "
        f"{len(sections)} sections → [bold]{len(all_docs)}[/bold] chunks  "
        f"(avg {avg} chars/chunk)"
    )
    return all_docs


def save_chunks_json(docs: list[Document], output_path: Path) -> None:
    """Persist chunks as JSON for inspection / reproducibility."""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    payload = [
        {"page_content": d.page_content, "metadata": d.metadata}
        for d in docs
    ]
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2),
                           encoding="utf-8")
    console.print(f"[dim]Chunks saved to {output_path}[/dim]")
