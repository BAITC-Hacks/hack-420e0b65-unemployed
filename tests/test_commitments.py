from datetime import date

import pytest

from minutes.extraction import validate_evidence
from minutes.models import Extraction, Segment

MEETING = date(2026, 9, 23)


def extract(texts, quote, cited=None, speaker=None):
    segments = [
        Segment(id=i, start=i * 3, end=i * 3 + 3, text=text, speaker=f"SPEAKER_0{i}")
        for i, text in enumerate(texts)
    ]
    value = Extraction(
        summary="Встреча",
        action_items=[
            {
                "task": "Задача",
                "responsible": None,
                "responsible_speaker": speaker,
                "deadline": None,
                "deadline_text": None,
                "evidence_segment_ids": cited or [0],
                "evidence_quote": quote,
            }
        ],
    )
    return validate_evidence(value, segments, {}, MEETING).action_items


@pytest.mark.parametrize(
    ("text", "quote"),
    [
        ("Ты придёшь завтра?", "Ты придёшь завтра?"),
        ("Ты придёшь на завтрашний хакатон?", "Ты придёшь на завтрашний хакатон"),  # "?" trimmed
        ("Может быть сделаем отчёт?", "Может быть сделаем отчёт?"),
        ("Может быть сделаем отчёт.", "сделаем отчёт"),
        ("Привет. Ты придёшь завтра?", "Ты придёшь завтра"),
        ("Мүмкін есепті дайындармыз.", "есепті дайындармыз"),
    ],
)
def test_questions_and_suggestions_are_not_action_items(text, quote):
    assert extract([text], quote) == []


def test_first_person_commitment_is_kept():
    items = extract(["Я приду завтра."], "Я приду завтра.", speaker="SPEAKER_00")
    assert len(items) == 1
    assert items[0].deadline == date(2026, 9, 24)


def test_imperative_assignment_is_kept():
    items = extract(["Сделай отчёт к пятнице."], "Сделай отчёт к пятнице.")
    assert len(items) == 1
    assert items[0].deadline == date(2026, 9, 25)


def test_accepted_suggestion_is_kept():
    texts = ["Может быть сделаем отчёт?", "Да, я сделаю."]
    assert len(extract(texts, "Может быть сделаем отчёт?", cited=[0, 1])) == 1


def test_question_answered_in_same_segment_is_kept():
    assert len(extract(["Сделаешь отчёт? Да, сделаю."], "Сделаешь отчёт?")) == 1


def test_unrelated_statement_is_not_acceptance():
    texts = ["Ты придёшь завтра?", "Погода хорошая сегодня."]
    assert extract(texts, "Ты придёшь завтра?", cited=[0, 1]) == []
