"""Tests for quiz history and weak-topic ranking, using fixed dates (no MCP involved)."""

from datetime import date, timedelta

import pytest

from study_mcp.quiz import QuizError, QuizLog

DAY0 = date(2026, 1, 1)


@pytest.fixture
def log(tmp_path):
    return QuizLog(tmp_path / "quiz.db")


def answer(log: QuizLog, topic: str, correct: bool, mistake: str = "", day: date = DAY0, q: str = "Q"):
    log.record(topic, q, correct, "", mistake, day)


def test_weakest_topic_comes_first(log):
    answer(log, "GCD", True)
    answer(log, "GCD", True)
    answer(log, "RSA", True)
    answer(log, "RSA", False, "wrong d", q="Find d")
    scores = log.topics(DAY0)
    assert [(s.topic, s.correct, s.attempts) for s in scores] == [("RSA", 1, 2), ("GCD", 2, 2)]
    assert scores[0].accuracy == 0.5
    assert scores[0].recent_mistakes == ["Find d (wrong d)"]


def test_topic_names_group_ignoring_case_and_spaces(log):
    answer(log, "Euclid's algorithm", True)
    answer(log, "  euclid's   ALGORITHM ", False)
    [score] = log.topics(DAY0)
    assert (score.topic, score.attempts) == ("Euclid's algorithm", 2)


def test_more_misses_rank_first_when_accuracy_ties(log):
    answer(log, "Mod", False)
    answer(log, "RSA", False)
    answer(log, "RSA", False)
    assert [s.topic for s in log.topics(DAY0)] == ["RSA", "Mod"]


def test_days_limits_to_recent_answers(log):
    answer(log, "RSA", False, day=DAY0)
    answer(log, "RSA", True, day=DAY0 + timedelta(days=10))
    [recent] = log.topics(DAY0 + timedelta(days=10), days=7)
    assert (recent.correct, recent.attempts, recent.recent_mistakes) == (1, 1, [])
    [all_time] = log.topics(DAY0 + timedelta(days=10))
    assert all_time.attempts == 2


def test_mistake_is_dropped_for_right_answers(log):
    answer(log, "RSA", True, mistake="ignored")
    answer(log, "RSA", False)
    assert log.topics(DAY0)[0].recent_mistakes == ["Q"]


def test_empty_topic_is_refused(log):
    with pytest.raises(QuizError):
        log.record("  ", "Q", True, "", "", DAY0)
