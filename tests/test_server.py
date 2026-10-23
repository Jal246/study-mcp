"""Tests that talk to the server through a real MCP client, the same way Claude does."""

from pathlib import Path

import pytest
from mcp import Client

from study_mcp import server
from study_mcp.server import mcp

pytestmark = pytest.mark.anyio

FIXTURE_NOTES = Path(__file__).parent / "fixtures" / "notes"


@pytest.fixture(autouse=True)
def use_fixture_notes(monkeypatch):
    """Point the server at the sample notes in tests/fixtures, not the user's real notes."""
    monkeypatch.setattr(server, "NOTES_DIR", FIXTURE_NOTES.resolve())


def text_of(result) -> str:
    return "\n".join(block.text for block in result.content)


async def call(tool: str, args: dict | None = None):
    async with Client(mcp) as client:
        return await client.call_tool(tool, args or {})


async def test_lists_all_tools():
    async with Client(mcp) as client:
        tools = await client.list_tools()
        assert {t.name for t in tools.tools} == {"list_notes", "read_note", "search_notes"}


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
