"""
Step 1 — EPUB → Markdown conversion.

Structure of this EPUB (Giorgio Bassani):
  - Spine items with class "p10"  → Part/Prologo/Epilogo title (no <h> tags)
  - Spine items whose first text  → section number ("1", "2", …)
    paragraph is only digits
  - Regular paragraphs            → prose content

We detect these patterns and emit proper Markdown headings:
  - Part titles  → ## Parte prima  (H2)
  - Section nums → ### Parte prima — 1  (H3, combining parent part + number)
  - Prose        → plain Markdown paragraphs
"""

import re
import html2text
import ebooklib
from ebooklib import epub
from bs4 import BeautifulSoup
from pathlib import Path
from rich.console import Console

console = Console()

# CSS classes used by this EPUB for structural elements
_PART_HEADING_CLASSES = {"p10", "p7"}   # part / prologo / epilogo titles


def _get_html2text() -> html2text.HTML2Text:
    h = html2text.HTML2Text()
    h.ignore_links    = True
    h.ignore_images   = True
    h.body_width      = 0
    h.unicode_snob    = True
    return h


def _para_class(tag) -> str:
    """Return normalised CSS class string for a <p> tag."""
    cls = tag.get("class", [])
    return " ".join(cls) if isinstance(cls, list) else str(cls)


def _is_part_heading(soup: BeautifulSoup) -> str | None:
    """
    Return the heading text if this spine item is a Part/Prologo/Epilogo
    title page, else None.
    """
    body = soup.find("body")
    if not body:
        return None
    # Find the first non-empty paragraph
    for p in body.find_all("p"):
        text = p.get_text().strip().strip("\xa0").strip()
        if not text:
            continue
        cls = _para_class(p)
        if any(c in cls for c in _PART_HEADING_CLASSES):
            return text
        break
    return None


def _extract_section_number(soup: BeautifulSoup) -> str | None:
    """
    If the first content paragraph of this spine item is just a digit string
    (e.g. '1', '2', '12'), return it as a string; else None.
    These are the intra-part section numbers Bassani uses.
    """
    body = soup.find("body")
    if not body:
        return None
    for p in body.find_all("p"):
        text = p.get_text().strip().strip("\xa0").strip()
        if not text:
            continue
        if re.fullmatch(r"\d{1,3}", text):
            return text
        break
    return None


def _html_to_md(html_content: str) -> str:
    """
    Convert HTML body to Markdown, stripping the first structural paragraph
    (part heading OR section number) so it isn't duplicated in the prose.
    """
    soup = BeautifulSoup(html_content, "lxml")
    body = soup.find("body")
    if not body:
        return ""

    # Remove the first non-empty paragraph if it is:
    #   a) a part/prologo heading  (p10 / p7 class), OR
    #   b) a bare section number   (only digits)
    for p in body.find_all("p"):
        text = p.get_text().strip().strip("\xa0").strip()
        if not text:
            continue
        cls = _para_class(p)
        is_heading = any(c in cls for c in _PART_HEADING_CLASSES)
        is_number  = bool(re.fullmatch(r"\d{1,3}", text))
        if is_heading or is_number:
            p.decompose()
        break

    h = _get_html2text()
    return h.handle(str(body)).strip()


def convert_epub_to_markdown(epub_path: Path, output_path: Path) -> Path:
    """
    Parse the EPUB spine in order, detect structural elements, and write a
    Markdown file where:
      - Part/Prologo/Epilogo titles become  ## headings  (H2)
      - In-part section numbers become      ### headings (H3)  e.g. ### Parte prima — 1
      - Prose content is plain Markdown paragraphs
    """
    output_path.parent.mkdir(parents=True, exist_ok=True)
    book = epub.read_epub(str(epub_path))

    console.print(f"[bold cyan]Converting EPUB:[/bold cyan] {epub_path.name}")

    sections: list[str] = []
    spine_ids = [item_id for item_id, _ in book.spine]

    current_part: str = "Prefazione"   # running Part context
    part_counter: int = 0
    section_counter: int = 0

    for item_id in spine_ids:
        item = book.get_item_with_id(item_id)
        if item is None or item.get_type() != ebooklib.ITEM_DOCUMENT:
            continue

        html_raw = item.get_content().decode("utf-8", errors="replace")
        soup     = BeautifulSoup(html_raw, "lxml")

        # ── Case 1: Part / Prologo / Epilogo title page ───────────────────────
        heading = _is_part_heading(soup)
        if heading:
            current_part    = heading
            part_counter   += 1
            section_counter = 0
            sections.append(f"## {heading}")
            # Some spine items (Prologo, Vita e morte di Micòl) carry BOTH
            # the heading AND body content in the same document.
            # Extract the body (heading paragraph already stripped by _html_to_md)
            # and append it as prose under this ## heading.
            body_md = _html_to_md(html_raw).strip()
            if body_md:
                sections.append(body_md)
            continue

        # ── Case 2: Numbered section (1, 2, 3 …) ──────────────────────────────
        sec_num = _extract_section_number(soup)
        body_md = _html_to_md(html_raw).strip()
        if not body_md:
            continue

        if sec_num:
            section_counter += 1
            # H3 encodes both the parent part and section number
            heading_md = f"### {current_part} — {sec_num}"
            sections.append(f"{heading_md}\n\n{body_md}")
        else:
            # Prefatory text before first ## heading (Montale's review, etc.)
            sections.append(body_md)

    full_text = "\n\n---\n\n".join(sections)

    output_path.write_text(full_text, encoding="utf-8")
    console.print(
        f"[green]Saved Markdown:[/green] {output_path}  "
        f"({len(full_text):,} chars, {len(sections)} sections, "
        f"{part_counter} parts, {section_counter} numbered sections)"
    )
    return output_path
