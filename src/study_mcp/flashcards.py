"""Flashcards with spaced repetition, stored in SQLite.

Spaced repetition: every time you answer a card, it is scheduled further into the future if you
knew it, and back to tomorrow if you didn't. Easy cards fade into the background; hard ones keep
coming back. The scheduling rule is SM-2, the algorithm Anki grew out of.

This module knows nothing about MCP, so it can be used and tested as plain Python.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

# Every card starts here. Each answer nudges it: up for easy answers, down for hard ones.
START_EASE = 2.5
# SM-2's floor: below this, cards you struggle with would come back almost every day forever.
MIN_EASE = 1.3
# Answers graded below this count as "didn't know it".
PASSING_QUALITY = 3

SCHEMA = """
CREATE TABLE IF NOT EXISTS cards (
    id            INTEGER PRIMARY KEY,
    question      TEXT    NOT NULL,
    answer        TEXT    NOT NULL,
    source        TEXT    NOT NULL DEFAULT '',  -- Where it came from, e.g. "slides.pptx (slide 2)".
    ease          REAL    NOT NULL,
    interval_days INTEGER NOT NULL DEFAULT 0,   -- Gap between the last review and the next one.
    repetitions   INTEGER NOT NULL DEFAULT 0,   -- Correct answers in a row.
    lapses        INTEGER NOT NULL DEFAULT 0,   -- Times it was forgotten.
    due           TEXT    NOT NULL,             -- ISO date, e.g. "2026-10-24".
    created       TEXT    NOT NULL
);
-- One row per answer, so we can show history ("reviewed today") without touching the cards.
CREATE TABLE IF NOT EXISTS reviews (
    id          INTEGER PRIMARY KEY,
    card_id     INTEGER NOT NULL REFERENCES cards(id) ON DELETE CASCADE,
    reviewed_on TEXT    NOT NULL,
    quality     INTEGER NOT NULL
);
CREATE INDEX IF NOT EXISTS cards_by_due ON cards(due);
"""


class CardError(Exception):
    """A problem the user should hear about, like an unknown card id."""


@dataclass(frozen=True)
class Card:
    id: int
    question: str
    answer: str
    source: str
    ease: float
    interval_days: int
    repetitions: int
    lapses: int
    due: date


def schedule(repetitions: int, interval_days: int, ease: float, quality: int) -> tuple[int, int, float]:
    """SM-2: given a card's state and how well it was just answered (0-5), return its new
    (repetitions, interval_days, ease).

    Right answers: the first gap is 1 day, the second 6, then each gap is the last one times ease.
    Wrong answers: the streak resets and the card is due again tomorrow.
    Either way, ease moves: +0.1 for a perfect answer, a bit less for 4, down for 3 and below.
    """
    if quality < PASSING_QUALITY:
        repetitions, interval_days = 0, 1
    else:
        if repetitions == 0:
            interval_days = 1
        elif repetitions == 1:
            interval_days = 6
        else:
            interval_days = round(interval_days * ease)
        repetitions += 1
    miss = 5 - quality
    ease = max(MIN_EASE, ease + 0.1 - miss * (0.08 + miss * 0.02))
    return repetitions, interval_days, ease


def _card(row: sqlite3.Row) -> Card:
    return Card(
        id=row["id"],
        question=row["question"],
        answer=row["answer"],
        source=row["source"],
        ease=row["ease"],
        interval_days=row["interval_days"],
        repetitions=row["repetitions"],
        lapses=row["lapses"],
        due=date.fromisoformat(row["due"]),
    )


class Deck:
    """All flashcards, in one SQLite file (created on first use)."""

    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        """Open the database for one operation: commit if it succeeds, roll back if it fails.

        (sqlite3's own `with connect(...)` commits but never closes the file, hence this helper.)
        """
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row  # Rows act like dicts: row["question"].
        conn.execute("PRAGMA foreign_keys = ON")  # SQLite ignores REFERENCES unless told not to.
        try:
            with conn:
                conn.executescript(SCHEMA)
                yield conn
        finally:
            conn.close()

    def _get(self, conn: sqlite3.Connection, card_id: int) -> Card:
        row = conn.execute("SELECT * FROM cards WHERE id = ?", (card_id,)).fetchone()
        if row is None:
            raise CardError(f"No card #{card_id}")
        return _card(row)

    def add(self, question: str, answer: str, source: str, today: date) -> Card:
        """Add a card, due today. Refuses a question that is already in the deck."""
        question, answer = question.strip(), answer.strip()
        if not question or not answer:
            raise CardError("question and answer must not be empty")
        with self._db() as conn:
            # The "?" placeholders let sqlite3 insert the values safely. Never build SQL with
            # f-strings: a question containing a quote would break it (SQL injection).
            existing = conn.execute(
                "SELECT id FROM cards WHERE lower(question) = lower(?)", (question,)
            ).fetchone()
            if existing:
                raise CardError(f"That question is already card #{existing['id']}")
            cursor = conn.execute(
                "INSERT INTO cards (question, answer, source, ease, due, created) VALUES (?, ?, ?, ?, ?, ?)",
                (question, answer, source.strip(), START_EASE, today.isoformat(), today.isoformat()),
            )
            return self._get(conn, cursor.lastrowid)

    def due(self, today: date, limit: int, source: str = "", cram: bool = False) -> list[Card]:
        """Cards due today or earlier, most overdue first.

        Args:
            source: Only cards whose source contains this text (any case), e.g. "Module 4".
            cram: Also include cards not due yet (soonest first), for reviewing before an exam.
        """
        # In LIKE, % and _ are wildcards; escape them so "10_22" means a literal underscore.
        pattern = "%" + source.strip().replace("\\", "\\\\").replace("%", "\\%").replace("_", "\\_") + "%"
        with self._db() as conn:
            rows = conn.execute(
                "SELECT * FROM cards WHERE (due <= ? OR ?) AND source LIKE ? ESCAPE '\\' "
                "ORDER BY due, id LIMIT ?",
                (today.isoformat(), cram, pattern, limit),
            ).fetchall()
            return [_card(row) for row in rows]

    def grade(self, card_id: int, quality: int, today: date) -> Card:
        """Record an answer (quality 0-5) and reschedule the card."""
        if not 0 <= quality <= 5:
            raise CardError(f"quality must be 0-5, got {quality}")
        with self._db() as conn:
            card = self._get(conn, card_id)
            repetitions, interval_days, ease = schedule(card.repetitions, card.interval_days, card.ease, quality)
            conn.execute(
                "UPDATE cards SET repetitions = ?, interval_days = ?, ease = ?, lapses = lapses + ?, due = ? "
                "WHERE id = ?",
                (
                    repetitions,
                    interval_days,
                    ease,
                    int(quality < PASSING_QUALITY),
                    (today + timedelta(days=interval_days)).isoformat(),
                    card_id,
                ),
            )
            conn.execute(
                "INSERT INTO reviews (card_id, reviewed_on, quality) VALUES (?, ?, ?)",
                (card_id, today.isoformat(), quality),
            )
            return self._get(conn, card_id)

    def delete(self, card_id: int) -> Card:
        """Remove a card and its review history."""
        with self._db() as conn:
            card = self._get(conn, card_id)
            conn.execute("DELETE FROM cards WHERE id = ?", (card_id,))
            return card

    def stats(self, today: date) -> dict:
        """Counts for a progress summary, plus the cards forgotten most often."""
        with self._db() as conn:
            counts = conn.execute(
                """
                SELECT count(*) AS total,
                       count(*) FILTER (WHERE due <= :today) AS due,
                       count(*) FILTER (WHERE id NOT IN (SELECT card_id FROM reviews)) AS new
                FROM cards
                """,
                {"today": today.isoformat()},
            ).fetchone()
            reviewed_today = conn.execute(
                "SELECT count(*) FROM reviews WHERE reviewed_on = ?", (today.isoformat(),)
            ).fetchone()[0]
            struggling = conn.execute(
                "SELECT * FROM cards WHERE lapses > 0 ORDER BY lapses DESC, ease LIMIT 5"
            ).fetchall()
            return {
                **dict(counts),
                "reviewed_today": reviewed_today,
                "struggling": [_card(row) for row in struggling],
            }
