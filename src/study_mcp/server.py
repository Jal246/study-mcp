"""Study Helper MCP server.

Gives an AI assistant (like Claude) tools to browse and search your study notes.
"""

import hashlib
import logging
import os
import re
import unicodedata
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import numpy as np
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
import pypdfium2 as pdfium

# Where the notes live. Override with the STUDY_NOTES_DIR environment variable.
DEFAULT_NOTES_DIR = Path(__file__).resolve().parents[2] / "notes"
NOTES_DIR = Path(os.environ.get("STUDY_NOTES_DIR", DEFAULT_NOTES_DIR)).resolve()

# File types we know how to read.
TEXT_EXTENSIONS = {".md", ".txt"}
PDF_EXTENSIONS = {".pdf"}
NOTE_EXTENSIONS = TEXT_EXTENSIONS | PDF_EXTENSIONS

# read_note stops after this many characters so one huge PDF can't flood the AI's context.
MAX_READ_CHARS = 40_000

# Semantic search: a small embedding model that runs locally (no API key, works offline after
# the first ~70 MB download). It turns text into a list of 384 numbers (a "vector") that captures
# its meaning, so "how to find the remainder" lands close to a slide about "n mod d".
EMBED_MODEL = "BAAI/bge-small-en-v1.5"
# The downloaded model and saved vectors live here. Override with STUDY_CACHE_DIR.
DEFAULT_CACHE_DIR = Path(__file__).resolve().parents[2] / ".cache"
CACHE_DIR = Path(os.environ.get("STUDY_CACHE_DIR", DEFAULT_CACHE_DIR)).resolve()
MODEL_DIR = CACHE_DIR / "models"
VECTOR_DIR = CACHE_DIR / "vectors"
# Notes are split into chunks of about this many characters; each chunk gets its own vector.
CHUNK_CHARS = 1000
# Similarity scores go from -1 to 1, but with this model even unrelated text scores about 0.5,
# while real matches score about 0.65 and up. Results below this are dropped.
MIN_SCORE = 0.6

# The model download logs every HTTP request; keep that noise out of the server's logs.
logging.getLogger("httpx").setLevel(logging.WARNING)
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

mcp = MCPServer(
    name="study-helper",
    instructions=(
        "Tools for the user's personal study notes (text files and PDF lecture slides). "
        "Use list_notes to see what exists. To find things, use semantic_search for questions and "
        "concepts in the user's own words (it matches meaning, not exact words), and search_notes "
        "for an exact term or formula. Use read_note to read a note (for PDFs you can read just "
        "some pages). When answering from notes, say which note and page the answer came from."
    ),
)


def _normalize(text: str) -> str:
    """Make text searchable.

    PDFs often store math as special Unicode letters: "𝑚𝑜𝑑" instead of "mod". They look the same
    but are different characters, so a search for "mod" would miss them. NFKC normalization turns
    those look-alikes back into plain letters.
    """
    return unicodedata.normalize("NFKC", text)


@lru_cache(maxsize=64)
def _load_pages(path: Path, modified_ns: int) -> list[str]:
    """Return a note's text as a list of pages (text files count as one page).

    PDF text extraction is slow, so results are cached. `modified_ns` (the file's last-modified
    time) is part of the cache key: when the file changes, the key changes and we re-read it.
    """
    if path.suffix.lower() in PDF_EXTENSIONS:
        with pdfium.PdfDocument(path) as pdf:
            return [_normalize(page.get_textpage().get_text_range()) for page in pdf]
    return [_normalize(path.read_text(encoding="utf-8"))]


def _pages(path: Path) -> list[str]:
    return _load_pages(path, path.stat().st_mtime_ns)


def _is_pdf(path: Path) -> bool:
    return path.suffix.lower() in PDF_EXTENSIONS


def _all_notes() -> list[Path]:
    """Every note file inside NOTES_DIR (including subfolders), sorted by path."""
    return sorted(
        p for p in NOTES_DIR.rglob("*") if p.is_file() and p.suffix.lower() in NOTE_EXTENSIONS
    )


def _name(path: Path) -> str:
    """A note's path relative to NOTES_DIR, with forward slashes, e.g. "math/module4.pdf"."""
    return path.relative_to(NOTES_DIR).as_posix()


def _safe_path(relative_path: str) -> Path:
    """Turn a user-given path into a real path, refusing anything outside NOTES_DIR.

    Without this check, a request like "../../secret.txt" could read any file on the computer.
    """
    path = (NOTES_DIR / relative_path).resolve()
    if not path.is_relative_to(NOTES_DIR):
        raise ToolError(f"'{relative_path}' is outside the notes folder")
    if not path.is_file():
        raise ToolError(f"No note found at '{relative_path}'. Use list_notes to see available notes.")
    return path


@mcp.tool()
def list_notes() -> str:
    """List all study notes, with their paths and sizes (page counts for PDFs)."""
    notes = _all_notes()
    if not notes:
        return f"No notes found in {NOTES_DIR}"
    lines = []
    for p in notes:
        detail = f"PDF, {len(_pages(p))} pages" if _is_pdf(p) else f"{p.stat().st_size} bytes"
        lines.append(f"{_name(p)} ({detail})")
    return f"{len(notes)} notes in {NOTES_DIR}:\n" + "\n".join(lines)


@mcp.tool()
def read_note(path: str, start_page: int | None = None, end_page: int | None = None) -> str:
    """Read the text of one note. For PDFs, optionally read only a range of pages.

    Args:
        path: The note's path as shown by list_notes, e.g. "math/module4.pdf".
        start_page: First page to read (PDFs only, starts at 1). Defaults to the first page.
        end_page: Last page to read, inclusive (PDFs only). Defaults to the last page.
    """
    note = _safe_path(path)
    pages = _pages(note)
    if not _is_pdf(note):
        return pages[0]

    first = start_page or 1
    last = end_page or len(pages)
    if not 1 <= first <= last <= len(pages):
        raise ToolError(f"Invalid page range {first}-{last}: '{path}' has pages 1-{len(pages)}")

    parts = []
    total = 0
    for number in range(first, last + 1):
        chunk = f"--- Page {number} ---\n{pages[number - 1].strip()}\n"
        if total + len(chunk) > MAX_READ_CHARS:
            parts.append(
                f"[Stopped at page {number - 1} to stay under {MAX_READ_CHARS} characters. "
                f"Call read_note again with start_page={number} to continue.]"
            )
            break
        parts.append(chunk)
        total += len(chunk)
    return "\n".join(parts)


@mcp.tool()
def search_notes(query: str, max_results: int = 10) -> str:
    """Search all notes (including PDFs) for a whole word or phrase, case-insensitive.

    Matches whole words only: "mod" finds "7 mod 3" but not "Module".

    Each result shows where the match is: "note:line" for text files, "note (page N)" for PDFs.

    Args:
        query: Text to look for, e.g. "chlorophyll".
        max_results: Maximum number of matching lines to return.
    """
    needle = _normalize(query).strip()
    if not needle:
        raise ToolError("query must not be empty")
    # (?<!\w) and (?!\w) mean "no letter/digit right before or after", i.e. whole words only.
    # re.escape makes characters like "(" or "+" in the query match literally.
    pattern = re.compile(rf"(?<!\w){re.escape(needle)}(?!\w)", re.IGNORECASE)

    matches = []
    for note in _all_notes():
        for page_no, page in enumerate(_pages(note), start=1):
            for line_no, line in enumerate(page.splitlines(), start=1):
                if not pattern.search(line):
                    continue
                where = f"{_name(note)} (page {page_no})" if _is_pdf(note) else f"{_name(note)}:{line_no}"
                matches.append(f"{where}: {line.strip()}")
                if len(matches) >= max_results:
                    return "\n".join(matches) + f"\n(stopped after {max_results} results)"

    return "\n".join(matches) if matches else f"No notes mention '{query}'."


@dataclass(frozen=True)
class Chunk:
    where: str  # Location to show the user: "note (page N)" for PDFs, "note:line" for text files.
    text: str


def _split(text: str, max_chars: int = CHUNK_CHARS) -> list[tuple[int, str]]:
    """Split text into pieces of about max_chars, cutting only between lines.

    Returns (line number the piece starts on, piece text) pairs, skipping blank pieces.
    """
    pieces = []
    current: list[str] = []
    size = 0
    start = 1
    for line_no, line in enumerate(text.splitlines(), start=1):
        if current and size + len(line) > max_chars:
            pieces.append((start, "\n".join(current)))
            current, size = [], 0
        if not current:
            start = line_no
        current.append(line)
        size += len(line) + 1
    if current:
        pieces.append((start, "\n".join(current)))
    return [(line_no, piece) for line_no, piece in pieces if piece.strip()]


def _chunks(path: Path) -> list[Chunk]:
    """Cut a note into chunks small enough to embed. PDF chunks never cross a page boundary."""
    name = _name(path)
    chunks = []
    for page_no, page in enumerate(_pages(path), start=1):
        for line_no, piece in _split(page):
            where = f"{name} (page {page_no})" if _is_pdf(path) else f"{name}:{line_no}"
            chunks.append(Chunk(where, piece))
    return chunks


@lru_cache(maxsize=1)
def _embedder():
    """Load the embedding model once (it is downloaded on first use).

    Imported here, not at the top, so the server starts instantly and only pays for loading the
    model when semantic search is actually used.
    """
    from fastembed import TextEmbedding

    return TextEmbedding(EMBED_MODEL, cache_dir=str(MODEL_DIR))


def _unit(vectors: np.ndarray) -> np.ndarray:
    """Scale vectors to length 1, so a dot product between two of them is their cosine similarity."""
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


@lru_cache(maxsize=64)
def _load_vectors(path: Path, modified_ns: int) -> tuple[list[Chunk], np.ndarray]:
    """Return a note's chunks and their vectors (a matrix with one row per chunk).

    Embedding is slow (seconds per PDF), so vectors are also saved to disk. The file name is a
    hash of the model name and the chunk texts: if the note or the model changes, the hash
    changes, and the old file is simply never looked up again.
    """
    chunks = _chunks(path)
    if not chunks:
        return [], np.empty((0, 0))
    fingerprint = "\n".join([EMBED_MODEL, *(c.where + "\n" + c.text for c in chunks)])
    saved = VECTOR_DIR / f"{hashlib.sha256(fingerprint.encode()).hexdigest()}.npy"
    if saved.exists():
        return chunks, np.load(saved)
    vectors = _unit(np.array(list(_embedder().embed([c.text for c in chunks]))))
    VECTOR_DIR.mkdir(parents=True, exist_ok=True)
    np.save(saved, vectors)
    return chunks, vectors


def _vectors(path: Path) -> tuple[list[Chunk], np.ndarray]:
    return _load_vectors(path, path.stat().st_mtime_ns)


@mcp.tool()
def semantic_search(query: str, max_results: int = 5) -> str:
    """Find the parts of the notes that are about what the query means, even if they use
    different words. E.g. "how to find the remainder" finds slides about "n mod d".

    Best for questions and concepts in your own words; for an exact word or formula, use
    search_notes. The first search can take a while (it downloads a small model and indexes
    the notes); later searches are fast.

    Each result shows a similarity score (higher = closer in meaning), where the passage is, and
    its start. Use read_note on that page to read the full context.

    Args:
        query: A question or description, e.g. "how do I check if a number is prime".
        max_results: Maximum number of passages to return.
    """
    question = _normalize(query).strip()
    if not question:
        raise ToolError("query must not be empty")
    # query_embed (not embed) adds the instruction prefix this model expects for search queries.
    query_vector = _unit(np.array(list(_embedder().query_embed([question])))[0])

    scored: list[tuple[float, Chunk]] = []
    for note in _all_notes():
        chunks, vectors = _vectors(note)
        if chunks:
            scored.extend(zip((vectors @ query_vector).tolist(), chunks))
    scored.sort(key=lambda pair: pair[0], reverse=True)

    results = []
    for score, chunk in scored[:max_results]:
        if score < MIN_SCORE:
            break
        snippet = " ".join(chunk.text.split())  # Collapse line breaks and runs of spaces.
        if len(snippet) > 300:
            snippet = snippet[:300] + "..."
        results.append(f"[{score:.2f}] {chunk.where}: {snippet}")
    return "\n\n".join(results) if results else f"Nothing in the notes seems related to '{query}'."
