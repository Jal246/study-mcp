"""Upcoming exams: name, date, and what each one covers, so studying can focus on what's next.

This module knows nothing about MCP, so it can be used and tested as plain Python.
"""

import sqlite3
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import date
from pathlib import Path

SCHEMA = """
CREATE TABLE IF NOT EXISTS exams (
    id      INTEGER PRIMARY KEY,
    name    TEXT NOT NULL,              -- E.g. "Discrete Math midterm".
    on_date TEXT NOT NULL,              -- ISO date, e.g. "2026-11-10".
    covers  TEXT NOT NULL DEFAULT ''    -- Notes or topics, e.g. "Module 3-4, RSA".
);
"""


class ExamError(Exception):
    """A problem the user should hear about, like a date in the past."""


@dataclass(frozen=True)
class Exam:
    id: int
    name: str
    on_date: date
    covers: str


def _exam(row: sqlite3.Row) -> Exam:
    return Exam(row["id"], row["name"], date.fromisoformat(row["on_date"]), row["covers"])


class ExamList:
    """All exams, in one SQLite file (created on first use)."""

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

    def add(self, name: str, on_date: str, covers: str, today: date) -> Exam:
        name = " ".join(name.split())
        if not name:
            raise ExamError("name must not be empty")
        try:
            day = date.fromisoformat(on_date.strip())
        except ValueError:
            raise ExamError(f"date must look like 2026-11-10, got {on_date!r}") from None
        if day < today:
            raise ExamError(f"{day.isoformat()} is in the past")
        with self._db() as conn:
            cursor = conn.execute(
                "INSERT INTO exams (name, on_date, covers) VALUES (?, ?, ?)",
                (name, day.isoformat(), covers.strip()),
            )
            return _exam(conn.execute("SELECT * FROM exams WHERE id = ?", (cursor.lastrowid,)).fetchone())

    def upcoming(self, today: date) -> list[Exam]:
        """Exams from today on, soonest first."""
        with self._db() as conn:
            rows = conn.execute(
                "SELECT * FROM exams WHERE on_date >= ? ORDER BY on_date, id", (today.isoformat(),)
            ).fetchall()
            return [_exam(row) for row in rows]

    def delete(self, exam_id: int) -> Exam:
        with self._db() as conn:
            row = conn.execute("SELECT * FROM exams WHERE id = ?", (exam_id,)).fetchone()
            if row is None:
                raise ExamError(f"No exam #{exam_id}")
            conn.execute("DELETE FROM exams WHERE id = ?", (exam_id,))
            return _exam(row)
