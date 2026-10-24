# study-mcp

[![tests](https://github.com/Jal246/study-mcp/actions/workflows/tests.yml/badge.svg)](https://github.com/Jal246/study-mcp/actions/workflows/tests.yml)

A Python [MCP](https://modelcontextprotocol.io) server that gives Claude access to my class notes, so I can ask questions about them, get quizzed, and review flashcards while I study.

Everything runs locally: notes never leave the computer except as the snippets Claude reads, and semantic search uses a small embedding model on the CPU (no API key).

## Demo

A real session with my discrete math slides (trimmed):

> **Me:** My discrete math midterm is Nov 10, it covers module 4. What should I study?
>
> **Claude:** *(calls `add_exam`, `weak_topics`)* Saved: Discrete Math midterm on 2026-11-10, in 18 days. So far you've quizzed on division, Euclid's algorithm, GCD/LCM, modular exponentiation and RSA, all right, so let's go wider: here's a harder RSA question from pages 25–28…
>
> **Me:** Read page 17.
>
> **Claude:** *(calls `read_note`)* Example 14: Compute [72(−65) + 211] mod 7. Example 15: What about 6^{17} mod 16? …
>
> **Me:** Review all my Module 4 flashcards, cram mode.
>
> **Claude:** *(calls `get_due_cards(source="Module 4", cram=True)`)* 7 cards. First one: for n = −23, d = 6, what are n div d and n mod d?

## What it can do

| Tool | What it's for |
|---|---|
| `list_notes` | See every note, with page/slide counts for PDFs and decks |
| `read_note` | Read a note, or just some pages of a PDF or slides of a deck |
| `search_notes` | Find an exact word, term or formula |
| `semantic_search` | Find notes by meaning ("how do I find a remainder?" finds the slide on `n mod d`) |
| `add_flashcard` | Save a question/answer card, with where it came from |
| `get_due_cards` | Cards due for review today; filter by note, or `cram` to review cards before they're due |
| `grade_card` | Record how well I remembered a card (0–5); schedules the next review with SM-2 |
| `delete_flashcard` | Remove a card |
| `card_stats` | How many cards, how many due, reviews done today |
| `record_quiz_answer` | Save each graded quiz answer by topic, with what went wrong |
| `weak_topics` | Quiz accuracy by topic, weakest first, with recent mistakes: what to practice next |
| `add_exam` / `upcoming_exams` / `delete_exam` | Exam dates and what they cover, so studying focuses on what's next |

**Supported notes:** Markdown (`.md`), text (`.txt`), PDF (`.pdf`) and PowerPoint (`.pptx`, including tables, grouped shapes and speaker notes).

**Math in PDFs:** superscripts and subscripts come out readable, LaTeX-style: `6^{17} mod 16`, `p_{1}^{α_{1}}`. This includes equations exported from PowerPoint or Word, whose small glyphs would otherwise show up as random Odia/Tamil/Telugu letters (see `scripts/gen_cambria_math.py` for why), and whose brackets are drawn as shapes rather than text (`gcd(x, y)` would read as `gcd x, y`).

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
- "What am I weakest at?" / "Give me practice on my weak topics"
- "My discrete math midterm is Nov 10, it covers modules 3 and 4" / "What should I study today?"

The first semantic search downloads the embedding model (~70 MB) and indexes your notes; after that it works offline, and only changed notes are re-indexed.

## Configuration

All optional, set as environment variables:

| Variable | Default | What it is |
|---|---|---|
| `STUDY_NOTES_DIR` | `notes/` | Where your notes live |
| `STUDY_CACHE_DIR` | `.cache/` | Downloaded model and saved search vectors (safe to delete; rebuilt automatically) |
| `STUDY_DATA_DIR` | `data/` | Flashcards, review history, quiz history and exams (SQLite). **This is your own work: back it up.** |

## How it works

```
Claude ──MCP (stdio)──▶ server.py ──┬─▶ notes/        .md .txt .pdf .pptx (read only)
                                    ├─▶ .cache/       embedding model + note vectors
                                    └─▶ data/*.db     flashcards, quiz history, exams (SQLite)
```

- **`server.py`** defines the tools with the official MCP Python SDK. Claude decides when to call them; the server's instructions tell it the study workflow (check weak topics before a quiz, cram before an exam, ask one flashcard at a time).
- **`flashcards.py`, `quiz.py`, `exams.py`** are small SQLite-backed classes, each with its own tests.

## Design decisions

- **MCP instead of a chatbot app.** The server only provides tools and data; Claude does the talking. Any MCP client (Claude Code, Claude Desktop) can use it, and there's no UI to build.
- **Local embeddings, not an API.** [fastembed](https://github.com/qdrant/fastembed) runs `BAAI/bge-small-en-v1.5` (~70 MB) on the CPU: free, offline and private. Notes are split into line-aligned chunks; vectors are saved per note and only rebuilt when a note's modified time changes. For a few hundred pages, a NumPy dot product beats running a vector database.
- **SQLite for my own data.** Flashcards, reviews and quiz answers must survive restarts and are tiny; SQLite is one file with no server, in the Python standard library.
- **SM-2 for spaced repetition.** SuperMemo's algorithm, which Anki grew out of: each grade (0–5) adjusts a card's ease and next interval, so easy cards come back rarely and hard ones often.
- **Repairing PDF math at the character level.** PowerPoint and Word export small equation glyphs as glyph numbers, which PDF readers show as Indic letters (`6ଵ଻` instead of 6¹⁷). The fix maps those glyph numbers back to characters using a table generated from the Cambria Math font, and only inside fonts that also contain math letters, so real Odia or Tamil text is left alone. Superscripts are detected from each character's position and marked LaTeX-style. Brackets drawn as shapes are recognized from their outlines (tall and thin; curved means a parenthesis; which side is heavier means opening or closing) and put back into the text.
- **Safe file access.** `read_note` rejects paths that escape the notes folder (`../`), so Claude can only see notes.

## Known limitations

- Scanned PDFs (images of text) have no text to read; there's no OCR.
- Square-root signs in PDF equations are drawn as shapes too and are still dropped: `⌊√N⌋` reads as `⌊N⌋`.
- Handwriting annotations in PDFs are ignored.

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
