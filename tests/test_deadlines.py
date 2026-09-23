from datetime import date

import pytest

from minutes.deadlines import find_deadline_phrase, resolve_deadline
from minutes.diarization import assign_speakers
from minutes.extraction import validate_evidence
from minutes.models import Extraction, Segment, SpeakerTurn, Transcript, Word

WEDNESDAY = date(2026, 9, 23)  # Hackathon day, a Wednesday.


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("в понедельник", date(2026, 9, 28)),
        ("до пятницы", date(2026, 9, 25)),
        ("к четвергу", date(2026, 9, 24)),
        ("в среду", date(2026, 9, 30)),  # Said on Wednesday: next week's Wednesday.
        ("в эту среду", date(2026, 9, 23)),
        ("в следующий понедельник", date(2026, 9, 28)),
        ("в следующую пятницу", date(2026, 10, 2)),
        ("на следующей неделе", date(2026, 10, 2)),  # End of next work week.
        ("на этой неделе", date(2026, 9, 25)),
        ("до конца недели", date(2026, 9, 25)),
        ("осы аптаның соңына дейін", date(2026, 9, 25)),
        ("завтра", date(2026, 9, 24)),
        ("послезавтра", date(2026, 9, 25)),
        ("через неделю", date(2026, 9, 30)),
        ("жұмаға дейін", date(2026, 9, 25)),
        ("дүйсенбіге дейін", date(2026, 9, 28)),
        ("ертең", date(2026, 9, 24)),
        ("келесі аптада", date(2026, 10, 2)),
        ("25 сентября", None),  # Absolute dates are left to the model.
        ("когда-нибудь", None),
        (None, None),
    ],
)
def test_relative_deadlines_resolve_from_meeting_date(text, expected):
    assert resolve_deadline(text, WEDNESDAY) == expected


def test_weekday_counts_from_meeting_date_not_today():
    assert resolve_deadline("в понедельник", date(2026, 9, 25)) == date(2026, 9, 28)
    assert resolve_deadline("в понедельник", date(2026, 9, 28)) == date(2026, 10, 5)


def test_environment_is_not_mistaken_for_wednesday():
    assert find_deadline_phrase("Проверим в среде разработки", WEDNESDAY) is None
    assert find_deadline_phrase("Сделаю отчёт к среде.", WEDNESDAY) == "к среде"
    assert find_deadline_phrase("Свяжусь до конца недели.", WEDNESDAY) == "до конца недели"


def action(**overrides):
    base = {
        "task": "Подготовить отчёт",
        "responsible": None,
        "responsible_speaker": None,
        "deadline": None,
        "deadline_text": None,
        "evidence_segment_ids": [0],
        "evidence_quote": "подготовлю отчёт",
    }
    return Extraction(summary="Отчёт", action_items=[{**base, **overrides}])


def test_model_weekday_error_is_corrected_in_code():
    segments = [Segment(id=0, start=0, end=3, text="Я подготовлю отчёт в понедельник.")]
    value = action(deadline=date(2026, 9, 29), deadline_text="в понедельник")
    item = validate_evidence(value, segments, {}, WEDNESDAY).action_items[0]
    assert item.deadline == date(2026, 9, 28)


def test_missing_deadline_text_recovered_from_quote():
    segments = [Segment(id=0, start=0, end=3, text="Я подготовлю отчёт на следующей неделе.")]
    value = action(evidence_quote="подготовлю отчёт на следующей неделе")
    item = validate_evidence(value, segments, {}, WEDNESDAY).action_items[0]
    assert item.deadline_text == "на следующей неделе"
    assert item.deadline == date(2026, 10, 2)


def test_unresolvable_deadline_keeps_model_value():
    segments = [Segment(id=0, start=0, end=3, text="Я подготовлю отчёт к 25 сентября.")]
    value = action(deadline=date(2026, 9, 25), deadline_text="к 25 сентября")
    assert validate_evidence(value, segments, {}, WEDNESDAY).action_items[0].deadline == date(
        2026, 9, 25
    )


def test_task_not_credited_to_adjacent_speaker():
    segments = [
        Segment(id=0, start=0, end=2, text="Кто сделает отчёт?", speaker="SPEAKER_00"),
        Segment(id=1, start=2, end=4, text="Я подготовлю отчёт.", speaker="SPEAKER_01"),
    ]
    names = {"SPEAKER_00": "Ерлан", "SPEAKER_01": "Айгүл"}
    # Model credited the question-asker, citing only the answer segment.
    value = action(responsible="Ерлан", responsible_speaker="SPEAKER_00", evidence_segment_ids=[1])
    item = validate_evidence(value, segments, names, WEDNESDAY).action_items[0]
    assert item.responsible_speaker is None
    assert item.responsible is None


def test_named_assignee_survives_speaker_correction():
    segments = [
        Segment(id=0, start=0, end=3, text="Айгүл подготовит отчёт.", speaker="SPEAKER_00"),
        Segment(id=1, start=3, end=4, text="Хорошо.", speaker="SPEAKER_01"),
    ]
    names = {"SPEAKER_01": "Айгүл"}
    value = action(
        responsible="Айгүл",
        responsible_speaker="SPEAKER_01",
        evidence_quote="подготовит отчёт",
    )
    item = validate_evidence(value, segments, names, WEDNESDAY).action_items[0]
    assert item.responsible_speaker is None
    assert item.responsible == "Айгүл"


def test_correct_speaker_link_is_kept():
    segments = [Segment(id=0, start=0, end=3, text="Я подготовлю отчёт.", speaker="SPEAKER_01")]
    value = action(responsible_speaker="SPEAKER_01")
    item = validate_evidence(value, segments, {"SPEAKER_01": "Айгүл"}, WEDNESDAY).action_items[0]
    assert (item.responsible_speaker, item.responsible) == ("SPEAKER_01", "Айгүл")


def transcript(words):
    return Transcript(
        segments=[Segment(id=0, start=0, end=6, text=" ".join(w.text for w in words), words=words)],
        language="ru",
        duration=6,
        device="cpu",
        model="test",
    )


def test_word_in_short_pause_joins_adjacent_turn_not_unknown():
    words = [
        Word(start=0.0, end=0.8, text="Я"),
        Word(start=0.8, end=1.6, text="подготовлю"),
        Word(start=1.7, end=2.0, text="отчёт."),  # Diarizer turn ended at 1.6.
        Word(start=3.0, end=3.5, text="Спасибо."),
    ]
    turns = [
        SpeakerTurn(start=0, end=1.6, speaker="SPEAKER_00"),
        SpeakerTurn(start=2.9, end=4, speaker="SPEAKER_01"),
    ]
    result = assign_speakers(transcript(words), turns)
    assert [(s.text, s.speaker) for s in result.segments] == [
        ("Я подготовлю отчёт.", "SPEAKER_00"),
        ("Спасибо.", "SPEAKER_01"),
    ]


def test_word_equidistant_between_speakers_stays_unknown():
    words = [Word(start=1.1, end=1.4, text="да")]
    turns = [
        SpeakerTurn(start=0, end=1.0, speaker="SPEAKER_00"),
        SpeakerTurn(start=1.5, end=3, speaker="SPEAKER_01"),
    ]
    assert assign_speakers(transcript(words), turns).segments[0].speaker is None


def test_overlapping_turns_assign_by_larger_share_per_word():
    words = [
        Word(start=0.0, end=1.0, text="Я"),
        Word(start=1.0, end=2.0, text="сделаю,"),
        Word(start=2.6, end=3.4, text="хорошо."),
    ]
    turns = [
        SpeakerTurn(start=0, end=2.2, speaker="SPEAKER_00"),
        SpeakerTurn(start=1.8, end=4, speaker="SPEAKER_01"),  # Overlaps 1.8–2.2.
    ]
    result = assign_speakers(transcript(words), turns)
    assert [(s.text, s.speaker) for s in result.segments] == [
        ("Я сделаю,", "SPEAKER_00"),
        ("хорошо.", "SPEAKER_01"),
    ]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("ертеңге дейін", date(2026, 9, 24)),
        ("Ертеңге дейін", date(2026, 9, 24)),
        ("ертең", date(2026, 9, 24)),
        ("бүгін", date(2026, 9, 23)),
        ("бүгінге дейін", date(2026, 9, 23)),
    ],
)
def test_kazakh_tomorrow_and_today(text, expected):
    assert resolve_deadline(text, WEDNESDAY) == expected


def test_evidence_wording_beats_copied_deadline_text():
    # Real failure: model copied "до шести" from another action and dated it today.
    segments = [
        Segment(
            id=0,
            start=0,
            end=4,
            text="Мен README бойынша таза ортада іске қосып көремін, ертеңге дейін.",
        )
    ]
    value = action(
        deadline=date(2026, 9, 23),
        deadline_text="до шести",
        evidence_quote="іске қосып көремін, ертеңге дейін",
    )
    item = validate_evidence(value, segments, {}, WEDNESDAY).action_items[0]
    assert item.deadline == date(2026, 9, 24)
    assert item.deadline_text == "ертеңге дейін"


def test_model_deadline_kept_when_its_wording_is_in_evidence():
    segments = [Segment(id=0, start=0, end=3, text="Я подготовлю отчёт до шести, не завтра.")]
    value = action(
        deadline=date(2026, 9, 23),
        deadline_text="до шести",
        evidence_quote="подготовлю отчёт до шести",
    )
    assert validate_evidence(value, segments, {}, WEDNESDAY).action_items[0].deadline == date(
        2026, 9, 23
    )
