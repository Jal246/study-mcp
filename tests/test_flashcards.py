"""Tests for the spaced-repetition logic and storage, using fixed dates (no MCP involved)."""

from datetime import date, timedelta

import pytest

from study_mcp.flashcards import MIN_EASE, START_EASE, CardError, Deck, schedule

DAY0 = date(2026, 1, 1)


@pytest.fixture
def deck(tmp_path):
    return Deck(tmp_path / "cards.db")


def intervals(qualities: list[int]) -> list[int]:
    """The gap (in days) after each answer, for a card answered with these qualities in a row."""
    repetitions, interval, ease = 0, 0, START_EASE
    gaps = []
    for quality in qualities:
        repetitions, interval, ease = schedule(repetitions, interval, ease, quality)
        gaps.append(interval)
    return gaps


def test_easy_card_gaps_grow_fast():
    # 45 * 2.9 = 130.5 in theory, but 2.9 is stored as 2.9000000000000004, so it rounds up.
    assert intervals([5, 5, 5, 5, 5]) == [1, 6, 16, 45, 131]


def test_barely_correct_answers_still_grow_gaps():
    # Each 3 lowers ease by only 0.14, so the gaps still roughly double.
    assert intervals([3, 3, 3, 3, 3]) == [1, 6, 13, 27, 52]


def test_often_forgotten_card_stays_on_short_gaps():
    # Four misses push ease to the 1.3 floor; once learned, its gaps grow by only 1.3x.
    assert intervals([1, 1, 1, 1, 3, 3, 3, 3]) == [1, 1, 1, 1, 1, 6, 8, 10]


def test_forgetting_resets_to_tomorrow():
    assert intervals([5, 5, 5, 1, 5, 5]) == [1, 6, 16, 1, 1, 6]


def test_ease_never_drops_below_minimum():
    _, _, ease = schedule(0, 0, MIN_EASE, 0)
    assert ease == MIN_EASE


def test_new_card_is_due_today(deck):
    card = deck.add("What does BFS use?", "A queue", "slides.pptx (slide 2)", DAY0)
    assert card.due == DAY0
    assert deck.due(DAY0, limit=10) == [card]


def test_grading_moves_card_out_of_today(deck):
    card = deck.add("What does BFS use?", "A queue", "", DAY0)
    graded = deck.grade(card.id, 5, DAY0)
    assert graded.due == DAY0 + timedelta(days=1)
    assert deck.due(DAY0, limit=10) == []
    assert deck.due(DAY0 + timedelta(days=1), limit=10) == [graded]


def test_overdue_cards_come_first(deck):
    late = deck.add("Q1", "A1", "", DAY0)
    deck.grade(late.id, 5, DAY0)  # Due on day 1.
    fresh = deck.add("Q2", "A2", "", DAY0 + timedelta(days=3))  # Due on day 3.
    assert [c.id for c in deck.due(DAY0 + timedelta(days=3), limit=10)] == [late.id, fresh.id]
    assert len(deck.due(DAY0 + timedelta(days=3), limit=1)) == 1


def test_duplicate_question_is_refused(deck):
    deck.add("What does BFS use?", "A queue", "", DAY0)
    with pytest.raises(CardError, match="already card #1"):
        deck.add("what does bfs use?", "Queue", "", DAY0)


def test_bad_input_is_refused(deck):
    with pytest.raises(CardError, match="must not be empty"):
        deck.add("  ", "A queue", "", DAY0)
    card = deck.add("Q", "A", "", DAY0)
    with pytest.raises(CardError, match="quality must be 0-5"):
        deck.grade(card.id, 7, DAY0)
    with pytest.raises(CardError, match="No card #99"):
        deck.grade(99, 5, DAY0)


def test_quotes_in_text_are_stored_safely(deck):
    # With SQL built from f-strings, the quote would end the string early and break the query.
    card = deck.add("What's Euclid's algorithm for?", "gcd'); DROP TABLE cards; --", "", DAY0)
    assert deck.due(DAY0, limit=10)[0].answer == "gcd'); DROP TABLE cards; --"
    assert card.id == 1


def test_stats_and_delete(deck):
    hard = deck.add("Hard one", "A", "", DAY0)
    deck.add("Never seen", "A", "", DAY0)
    deck.grade(hard.id, 1, DAY0)
    deck.grade(hard.id, 0, DAY0)

    stats = deck.stats(DAY0)
    assert (stats["total"], stats["due"], stats["new"], stats["reviewed_today"]) == (2, 1, 1, 2)
    assert [c.question for c in stats["struggling"]] == ["Hard one"]

    deck.delete(hard.id)  # Its reviews go with it (ON DELETE CASCADE).
    stats = deck.stats(DAY0)
    assert (stats["total"], stats["reviewed_today"], stats["struggling"]) == (1, 0, [])


def test_due_filters_by_source_and_cram_adds_cards_not_due(deck):
    math = deck.add("Q1", "A", "Module 4 (10_22).pdf (page 9)", DAY0)
    deck.add("Q2", "A", "biology/photosynthesis.md", DAY0)
    deck.grade(math.id, 5, DAY0)  # Now due tomorrow.
    assert [c.question for c in deck.due(DAY0, 10)] == ["Q2"]
    assert deck.due(DAY0, 10, source="module 4") == []
    assert [c.question for c in deck.due(DAY0, 10, source="module 4", cram=True)] == ["Q1"]
    assert deck.due(DAY0, 10, source="10%22", cram=True) == []  # % is literal, not a wildcard.
