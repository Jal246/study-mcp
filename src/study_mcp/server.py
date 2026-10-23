"""Study Helper MCP server.

Gives an AI assistant (like Claude) tools to browse and search your study notes.
"""

import os
import re
import unicodedata
from functools import lru_cache
from pathlib import Path

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

mcp = MCPServer(
    name="study-helper",
    instructions=(
        "Tools for the user's personal study notes (text files and PDF lecture slides). "
        "Use list_notes to see what exists, search_notes to find a topic, and read_note to read "
        "a note (for PDFs you can read just some pages). When answering from notes, say which "
        "note and page the answer came from."
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
