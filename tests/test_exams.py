"""Tests for saving and listing exams, using fixed dates (no MCP involved)."""

from datetime import date

import pytest

from study_mcp.exams import ExamError, ExamList

TODAY = date(2026, 10, 23)


@pytest.fixture
def exams(tmp_path):
    return ExamList(tmp_path / "exams.db")


def test_upcoming_is_soonest_first_and_skips_past_exams(exams):
    exams.add("Final", "2026-12-10", "everything", TODAY)
    exams.add("Midterm", "2026-11-10", "Module 3-4", TODAY)
    exams.add("Quiz", "2026-10-25", "", TODAY)
    assert [e.name for e in exams.upcoming(TODAY)] == ["Quiz", "Midterm", "Final"]
    assert [e.name for e in exams.upcoming(date(2026, 11, 11))] == ["Final"]


def test_bad_or_past_dates_are_refused(exams):
    with pytest.raises(ExamError, match="must look like"):
        exams.add("Midterm", "Nov 10", "", TODAY)
    with pytest.raises(ExamError, match="in the past"):
        exams.add("Midterm", "2026-10-01", "", TODAY)


def test_delete(exams):
    exam = exams.add("Midterm", "2026-11-10", "", TODAY)
    assert exams.delete(exam.id).name == "Midterm"
    assert exams.upcoming(TODAY) == []
    with pytest.raises(ExamError):
        exams.delete(exam.id)
