"""Section-aware chunker.

Each section is split into chunks of ~`max_chars` with `overlap_chars` of
overlap. Abstract and conclusion are kept as single chunks (whole or split if
they exceed the size cap) so retrieval can prefer them when the question is
about a paper's gist.
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from assistant.rag.parser import ParsedPaper, Section


@dataclass
class Chunk:
    id: str
    paper_id: str
    section_type: str
    section_title: str
    order: int
    text: str
    char_start: int
    char_end: int


def chunk_paper(
    paper_id: str,
    parsed: ParsedPaper,
    *,
    max_chars: int = 900,
    overlap_chars: int = 135,
) -> list[Chunk]:
    out: list[Chunk] = []
    order = 0
    for section in parsed.sections:
        if section.section_type in {"references", "acknowledgments"}:
            continue
        for text, (cs, ce) in _split(section, max_chars, overlap_chars):
            out.append(
                Chunk(
                    id=str(uuid.uuid4()),
                    paper_id=paper_id,
                    section_type=section.section_type,
                    section_title=section.title,
                    order=order,
                    text=text,
                    char_start=cs,
                    char_end=ce,
                )
            )
            order += 1
    return out


def _split(section: Section, max_chars: int, overlap: int) -> list[tuple[str, tuple[int, int]]]:
    text = section.text
    if len(text) <= max_chars:
        return [(text, (section.char_start, section.char_end))]
    pieces: list[tuple[str, tuple[int, int]]] = []
    step = max_chars - overlap
    i = 0
    while i < len(text):
        end = min(i + max_chars, len(text))
        piece = text[i:end]
        pieces.append((piece, (section.char_start + i, section.char_start + end)))
        if end == len(text):
            break
        i += step
    return pieces
