"""
Step 2 — Noise removal & text normalisation.

Cleans the raw Markdown that comes out of the EPUB converter:
- Strips page numbers, running headers/footers, project gutenberg boilerplate
- Normalises whitespace, dashes, and curly quotes → straight equivalents
- Removes empty Markdown artefacts (orphan ##, lone ***, etc.)
- Detects and reports potential noise patterns for manual review
"""

import re
from pathlib import Path
from rich.console import Console
from rich.table import Table

console = Console()


# ── Patterns to delete entirely ──────────────────────────────────────────────
_STRIP_PATTERNS: list[tuple[str, str]] = [
    # bare page numbers: only 3-4 digit standalone numbers (real page numbers).
    # 1-2 digit standalone numbers are section headings → NOT stripped here
    # (the converter already turns them into ### headings before this runs).
    (r"(?m)^\s*\d{3,4}\s*$",              "bare page numbers"),
    # common epub artefacts: lines like "* * *" or "— — —"
    (r"(?m)^\s*[\*\-–—]{1,3}(\s+[\*\-–—]{1,3}){2,}\s*$", "decorative separators"),
    # HTML entities that html2text sometimes leaves
    (r"&amp;|&nbsp;|&lt;|&gt;|&quot;",    "HTML entities"),
    # Calibre/ebooklib artefacts: [if gte mso …] style comments
    (r"\[if [^\]]+\].*?\[endif\]",         "conditional comments"),
    # lines that are just whitespace or underscores
    (r"(?m)^\s*[_\s]+$",                  "blank/underscore lines"),
]

# ── Normalisation replacements ────────────────────────────────────────────────
_NORMALISE_PATTERNS: list[tuple[str, str, str]] = [
    # curly quotes → straight (keeps Italian apostrophes readable)
    (r"[\u2018\u2019]",  "'",  "curly single quotes"),
    (r"[\u201c\u201d]",  '"',  "curly double quotes"),
    # em/en dashes used as dialogue markers → standard "—" (keep as-is, just unify)
    (r"\u2013",          "–",  "en-dash normalised"),
    # multiple spaces → single space
    (r"[ \t]{2,}",       " ",  "multiple spaces"),
    # more than 2 consecutive blank lines → exactly 2
    (r"\n{3,}",          "\n\n", "excess blank lines"),
]


def clean_markdown(text: str, verbose: bool = True) -> str:
    """Apply all cleaning rules and return sanitised text."""
    stats: dict[str, int] = {}

    # 1. Strip noise
    for pattern, label in _STRIP_PATTERNS:
        cleaned, n = re.subn(pattern, "", text, flags=re.DOTALL)
        if n:
            stats[label] = n
        text = cleaned

    # 2. Normalise
    for pattern, replacement, label in _NORMALISE_PATTERNS:
        cleaned, n = re.subn(pattern, replacement, text)
        if n:
            stats[label] = n
        text = cleaned

    # 3. Remove orphan Markdown headings (## followed immediately by ---)
    text = re.sub(r"(?m)^(#{1,4}\s*)\n+---", "---", text)

    # 4. Final strip
    text = text.strip()

    if verbose and stats:
        table = Table(title="[bold]Preprocessing — noise removed[/bold]",
                      show_header=True)
        table.add_column("Pattern", style="cyan")
        table.add_column("Occurrences", style="magenta", justify="right")
        for label, count in stats.items():
            table.add_row(label, str(count))
        console.print(table)

    return text


def preprocess_file(input_path: Path, output_path: Path | None = None) -> Path:
    """
    Read a Markdown file, clean it, and write back.
    If *output_path* is None, overwrites *input_path* in-place.
    """
    output_path = output_path or input_path
    raw = input_path.read_text(encoding="utf-8")

    console.print(f"[bold cyan]Preprocessing:[/bold cyan] {input_path.name}  "
                  f"({len(raw):,} chars before cleaning)")

    cleaned = clean_markdown(raw, verbose=True)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(cleaned, encoding="utf-8")

    reduction = 100 * (1 - len(cleaned) / len(raw))
    console.print(f"[green]Cleaned file saved:[/green] {output_path}  "
                  f"({len(cleaned):,} chars, -{reduction:.1f}% noise)")
    return output_path
