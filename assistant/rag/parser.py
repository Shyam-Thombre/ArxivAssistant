"""PDF -> structured sections.

MVP uses PyMuPDF + a heuristic section detector. Marker can be plugged in
later as a higher-quality parser by replacing `parse_pdf()` — the rest of the
pipeline only cares about the returned `ParsedPaper` shape.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

# Common section headers we want to recognize in ML/CS papers.
_SECTION_KEYWORDS = {
    "abstract": "abstract",
    "introduction": "intro",
    "background": "background",
    "related work": "related",
    "method": "method",
    "methods": "method",
    "methodology": "method",
    "approach": "method",
    "experiment": "experiments",
    "experiments": "experiments",
    "evaluation": "experiments",
    "results": "results",
    "discussion": "discussion",
    "analysis": "analysis",
    "conclusion": "conclusion",
    "conclusions": "conclusion",
    "limitations": "limitations",
    "references": "references",
    "acknowledgments": "acknowledgments",
    "acknowledgements": "acknowledgments",
    "appendix": "appendix",
}

_HEADING_RE = re.compile(
    r"^\s*(?:\d+(?:\.\d+)*\.?\s+)?([A-Za-z][A-Za-z \-/&]{2,60})\s*$"
)


@dataclass
class Section:
    title: str
    section_type: str  # one of _SECTION_KEYWORDS' values, or 'body'
    text: str
    char_start: int
    char_end: int


@dataclass
class ParsedPaper:
    full_text: str
    sections: list[Section]
    abstract: str | None


def parse_pdf(pdf_path: Path) -> ParsedPaper:
    import fitz  # PyMuPDF

    doc = fitz.open(pdf_path)
    full = "\n".join(page.get_text("text") for page in doc)
    doc.close()
    return _structure(full)


def _classify(line: str) -> str | None:
    m = _HEADING_RE.match(line)
    if not m:
        return None
    key = m.group(1).strip().lower()
    return _SECTION_KEYWORDS.get(key)


def _structure(full: str) -> ParsedPaper:
    lines = full.split("\n")
    sections: list[Section] = []
    cur_title = "Header"
    cur_type = "header"
    cur_start = 0
    cur_lines: list[str] = []
    cursor = 0

    def flush(end_pos: int) -> None:
        if not cur_lines:
            return
        text = "\n".join(cur_lines).strip()
        if text:
            sections.append(
                Section(
                    title=cur_title,
                    section_type=cur_type,
                    text=text,
                    char_start=cur_start,
                    char_end=end_pos,
                )
            )

    for raw in lines:
        line_len = len(raw) + 1  # +1 for the newline
        sec_type = _classify(raw.strip()) if raw.strip() else None
        if sec_type is not None:
            flush(cursor)
            cur_title = raw.strip()
            cur_type = sec_type
            cur_start = cursor
            cur_lines = []
        else:
            cur_lines.append(raw)
        cursor += line_len
    flush(cursor)

    abstract = None
    for s in sections:
        if s.section_type == "abstract":
            abstract = s.text
            break
    return ParsedPaper(full_text=full, sections=sections, abstract=abstract)
