"""Quiz history: every quiz answer, by topic, so we can find what the user is weakest at.

Flashcards remember single facts; this remembers how the user does on whole topics ("RSA",
"Euclid's algorithm") across quizzes, so practice can go where it's needed.

This module knows nothing about MCP, so it can be used and tested as plain Python.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date, timedelta
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS answers (
    id          INTEGER PRIMARY KEY,
    answered_on TEXT    NOT NULL,             -- ISO date, e.g. "2026-10-23".
    topic       TEXT    NOT NULL,             -- As first written, e.g. "RSA".
    topic_key   TEXT    NOT NULL,             -- Lowercased, so "rsa" and "RSA" group together.
    question    TEXT    NOT NULL,
    correct     INTEGER NOT NULL,             -- 1 or 0.
    source      TEXT    NOT NULL DEFAULT '',  -- Where the question came from.
    mistake     TEXT    NOT NULL DEFAULT ''   -- What went wrong, for wrong answers.
);
CREATE INDEX IF NOT EXISTS answers_by_topic ON answers(topic_key);
"""


class QuizError(Exception):
    """A problem the user should hear about, like an empty topic."""


@dataclass(frozen=True)
class TopicScore:
    topic: str
    attempts: int
    correct: int
    last_answered: date
    recent_mistakes: list[str]

    @property
    def accuracy(self) -> float:
        return self.correct / self.attempts


class QuizLog:
    """All quiz answers, in one SQLite file (created on first use)."""

    def __init__(self, path: Path):
        self.path = path

    @contextmanager
    def _db(self) -> Iterator[sqlite3.Connection]:
        """Open the database for one operation: commit if it succeeds, roll back if it fails."""
        self.path.parent.mkdir(parents=True, exist_ok=True)
        conn = sqlite3.connect(self.path)
        conn.row_factory = sqlite3.Row
        try:
            with conn:
                conn.executescript(SCHEMA)
                yield conn
        finally:
            conn.close()

    def record(
        self, topic: str, question: str, correct: bool, source: str, mistake: str, today: date
    ) -> int:
        """Save one answer; returns its id."""
        topic, question = " ".join(topic.split()), question.strip()
        if not topic or not question:
            raise QuizError("topic and question must not be empty")
        with self._db() as conn:
            # Keep the spelling the topic was first saved with, so the list doesn't show both
            # "RSA" and "rsa".
            first = conn.execute(
                "SELECT topic FROM answers WHERE topic_key = ? ORDER BY id LIMIT 1", (topic.lower(),)
            ).fetchone()
            cursor = conn.execute(
                "INSERT INTO answers (answered_on, topic, topic_key, question, correct, source, mistake) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (
                    today.isoformat(),
                    first["topic"] if first else topic,
                    topic.lower(),
                    question,
                    int(correct),
                    source.strip(),
                    "" if correct else mistake.strip(),
                ),
            )
            return cursor.lastrowid

    def topics(self, today: date, days: int | None = None) -> list[TopicScore]:
        """Every topic's score, weakest first (lowest accuracy, then most attempts).

        Args:
            days: Only count answers from the last this-many days; None counts everything.
        """
        since = (today - timedelta(days=days)).isoformat() if days else ""
        with self._db() as conn:
            rows = conn.execute(
                """
                SELECT topic_key, min(topic) AS topic, count(*) AS attempts,
                       sum(correct) AS correct, max(answered_on) AS last_answered
                FROM answers WHERE answered_on >= ?
                GROUP BY topic_key
                ORDER BY 1.0 * sum(correct) / count(*), count(*) DESC, topic_key
                """,
                (since,),
            ).fetchall()
            scores = []
            for row in rows:
                mistakes = conn.execute(
                    "SELECT question, mistake FROM answers "
                    "WHERE topic_key = ? AND answered_on >= ? AND correct = 0 "
                    "ORDER BY id DESC LIMIT 3",
                    (row["topic_key"], since),
                ).fetchall()
                scores.append(TopicScore(
                    topic=row["topic"],
                    attempts=row["attempts"],
                    correct=row["correct"],
                    last_answered=date.fromisoformat(row["last_answered"]),
                    recent_mistakes=[
                        f"{m['question']} ({m['mistake']})" if m["mistake"] else m["question"]
                        for m in mistakes
                    ],
                ))
            return scores
