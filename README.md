# study-mcp

A Python [MCP](https://modelcontextprotocol.io) server that gives Claude access to my class notes, so I can ask questions about them, get quizzed, and review flashcards while I study.

Everything runs locally: notes never leave the computer except as the snippets Claude reads, and semantic search uses a small embedding model on the CPU (no API key).

## What it can do

| Tool | What it's for |
|---|---|
| `list_notes` | See every note, with page/slide counts for PDFs and decks |
| `read_note` | Read a note, or just some pages of a PDF or slides of a deck |
| `search_notes` | Find an exact word, term or formula |
| `semantic_search` | Find notes by meaning ("how do I find a remainder?" finds the slide on `n mod d`) |
| `add_flashcard` | Save a question/answer card, with where it came from |
| `get_due_cards` | Cards due for review today |
| `grade_card` | Record how well I remembered a card (0–5); schedules the next review with SM-2 |
| `delete_flashcard` | Remove a card |
| `card_stats` | How many cards, how many due, reviews done today |

**Supported notes:** Markdown (`.md`), text (`.txt`), PDF (`.pdf`) and PowerPoint (`.pptx`, including tables, grouped shapes and speaker notes).

**Math in PDFs:** superscripts and subscripts come out readable, LaTeX-style: `6^{17} mod 16`, `p_{1}^{α_{1}}`. This includes equations exported from PowerPoint or Word, whose small glyphs would otherwise show up as random Odia/Tamil/Telugu letters (see `scripts/gen_cambria_math.py` for why).

## Setup

Needs [uv](https://docs.astral.sh/uv/) and Python 3.13+.

```sh
git clone https://github.com/Jal246/study-mcp.git
cd study-mcp
uv sync
```

Put your notes in `notes/` (subfolders are fine, e.g. `notes/math/module4.pdf`). That folder is git-ignored, so course material is never committed.

Register the server with Claude Code, using the full path to the repo:

```sh
claude mcp add study-helper -- uv run --directory /path/to/study-mcp study-mcp
```

Then ask Claude things like:

- "What notes do I have?"
- "Explain Euclid's algorithm from my number theory slides"
- "Give me 5 questions from the PDF and grade my answers"
- "Make flashcards from module 4" / "Quiz me"

The first semantic search downloads the embedding model (~70 MB) and indexes your notes; after that it works offline, and only changed notes are re-indexed.

## Configuration

All optional, set as environment variables:

| Variable | Default | What it is |
|---|---|---|
| `STUDY_NOTES_DIR` | `notes/` | Where your notes live |
| `STUDY_CACHE_DIR` | `.cache/` | Downloaded model and saved search vectors (safe to delete; rebuilt automatically) |
| `STUDY_DATA_DIR` | `data/` | Flashcards and review history (SQLite). **This is your own work: back it up.** |

## Development

```sh
uv run pytest
```

Tests run against sample notes in `tests/fixtures/notes`, never your real ones. `tests/fixtures/make_slides.py` rebuilds the sample `.pptx`.

`src/study_mcp/cambria_math.py` is generated. To rebuild it (Windows with Office fonts):

```sh
pip install fonttools
python scripts/gen_cambria_math.py
```
