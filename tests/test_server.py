"""Tests that talk to the server through a real MCP client, the same way Claude does."""

from pathlib import Path

import pytest
from mcp import Client

from study_mcp import server
from study_mcp.server import mcp

pytestmark = pytest.mark.anyio

FIXTURE_NOTES = Path(__file__).parent / "fixtures" / "notes"


@pytest.fixture(autouse=True)
def use_fixture_notes(monkeypatch, tmp_path):
    """Point the server at the sample notes in tests/fixtures, not the user's real notes.

    Saved vectors go to a temporary folder too (the downloaded model is still shared, so it is
    only downloaded once).
    """
    monkeypatch.setattr(server, "NOTES_DIR", FIXTURE_NOTES.resolve())
    monkeypatch.setattr(server, "VECTOR_DIR", tmp_path / "vectors")
    server._load_vectors.cache_clear()


def text_of(result) -> str:
    return "\n".join(block.text for block in result.content)


async def call(tool: str, args: dict | None = None):
    async with Client(mcp) as client:
        return await client.call_tool(tool, args or {})


async def test_lists_all_tools():
    async with Client(mcp) as client:
        tools = await client.list_tools()
        assert {t.name for t in tools.tools} == {
            "list_notes",
            "read_note",
            "search_notes",
            "semantic_search",
        }


async def test_list_notes():
    out = text_of(await call("list_notes"))
    assert "biology/photosynthesis.md" in out
    assert "cs/big-o.md" in out
    assert "lecture.pdf (PDF, 3 pages)" in out


async def test_read_note():
    out = text_of(await call("read_note", {"path": "cs/big-o.md"}))
    assert "binary search" in out


async def test_read_whole_pdf_marks_pages():
    out = text_of(await call("read_note", {"path": "lecture.pdf"}))
    assert "--- Page 1 ---" in out
    assert "--- Page 3 ---" in out
    assert "RSA encryption" in out


async def test_read_pdf_page_range():
    out = text_of(await call("read_note", {"path": "lecture.pdf", "start_page": 2, "end_page": 2}))
    assert "Euclid" in out
    assert "Page 1" not in out
    assert "RSA" not in out


async def test_read_pdf_invalid_page_range():
    result = await call("read_note", {"path": "lecture.pdf", "start_page": 5})
    assert result.is_error
    assert "has pages 1-3" in text_of(result)


async def test_search_is_case_insensitive_and_shows_location():
    out = text_of(await call("search_notes", {"query": "CHLOROPHYLL"}))
    assert "biology/photosynthesis.md:8:" in out


async def test_search_pdf_shows_page_number():
    out = text_of(await call("search_notes", {"query": "euclid"}))
    assert "lecture.pdf (page 2): Euclid's Algorithm" in out


async def test_search_finds_math_italic_letters():
    # The note contains "𝑚𝑜𝑑" (math italic letters, as PDFs store them); a plain "mod" must find it.
    out = text_of(await call("search_notes", {"query": "7 mod 3"}))
    assert "cs/math-letters.md:3:" in out


async def test_search_matches_whole_words_only():
    out = text_of(await call("search_notes", {"query": "mod"}))
    assert "Modular" not in out
    assert "lecture.pdf (page 2): gcd(y, x): repeat r := y mod x until r = 0" in out


async def test_search_respects_max_results():
    out = text_of(await call("search_notes", {"query": "mod", "max_results": 2}))
    assert "(stopped after 2 results)" in out


async def test_search_with_no_matches():
    out = text_of(await call("search_notes", {"query": "quantum"}))
    assert "No notes mention" in out


async def test_cannot_read_outside_notes_folder():
    result = await call("read_note", {"path": "../../pyproject.toml"})
    assert result.is_error
    assert "outside the notes folder" in text_of(result)


def first_result(out: str) -> str:
    return out.split("\n\n")[0]


async def test_semantic_search_matches_meaning_not_words():
    # None of these words appear in the RSA slide ("RSA encryption Uses modular exponentiation...").
    out = text_of(await call("semantic_search", {"query": "secret messages"}))
    assert "lecture.pdf (page 3): RSA encryption" in first_result(out)


async def test_semantic_search_remainder_finds_mod():
    out = text_of(await call("semantic_search", {"query": "how do I find the remainder after dividing"}))
    assert "mod" in first_result(out)
    assert "photosynthesis" not in out


async def test_semantic_search_text_note_shows_line():
    out = text_of(await call("semantic_search", {"query": "how do plants make food"}))
    assert "] biology/photosynthesis.md:1: # Photosynthesis" in first_result(out)


async def test_semantic_search_unrelated_query_finds_nothing():
    out = text_of(await call("semantic_search", {"query": "medieval castle architecture"}))
    assert "Nothing in the notes seems related" in out


async def test_semantic_search_respects_max_results():
    out = text_of(await call("semantic_search", {"query": "number theory", "max_results": 2}))
    assert out.count("\n\n") == 1


async def test_semantic_search_empty_query():
    result = await call("semantic_search", {"query": "   "})
    assert result.is_error
    assert "must not be empty" in text_of(result)


async def test_semantic_search_saves_vectors_and_reuses_them():
    await call("semantic_search", {"query": "gcd"})
    saved = sorted(server.VECTOR_DIR.glob("*.npy"))
    assert len(saved) == 4  # One file per note.
    modified = [p.stat().st_mtime_ns for p in saved]

    server._load_vectors.cache_clear()  # Forget the in-memory copy, like a server restart.
    await call("semantic_search", {"query": "gcd"})
    assert sorted(server.VECTOR_DIR.glob("*.npy")) == saved
    assert [p.stat().st_mtime_ns for p in saved] == modified  # Loaded from disk, not re-embedded.


def test_split_cuts_between_lines_and_keeps_line_numbers():
    text = "aaaa\nbbbb\n\ncccc\ndddd"
    assert server._split(text, max_chars=9) == [(1, "aaaa\nbbbb"), (3, "\ncccc"), (5, "dddd")]
    assert server._split("short\nnote") == [(1, "short\nnote")]
    assert server._split("\n\n  \n") == []
