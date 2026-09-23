from datetime import date

import pytest

from minutes.models import ActionItem
from minutes.review import apply_action_edits


def test_review_preserves_evidence_and_clears_incorrect_person_link():
    item = ActionItem(
        task="Отчёт",
        responsible="Айгүл",
        responsible_speaker="SPEAKER_00",
        deadline=None,
        deadline_text=None,
        evidence_segment_ids=[0],
        evidence_quote="Отчёт",
    )
    edited = apply_action_edits(
        [item],
        [
            {
                "Responsible": "Дана",
                "Task": "Проверить отчёт",
                "Deadline": date(2026, 9, 25),
                "Status": "completed",
            }
        ],
    )[0]
    assert edited.responsible == "Дана"
    assert edited.responsible_speaker is None
    assert edited.evidence_quote == "Отчёт"
    assert edited.status == "completed"
    assert item.responsible == "Айгүл"


def test_invalid_manual_status_is_rejected():
    item = ActionItem(
        task="Отчёт",
        responsible=None,
        responsible_speaker=None,
        deadline=None,
        deadline_text=None,
        evidence_segment_ids=[0],
        evidence_quote="Отчёт",
    )
    with pytest.raises(ValueError):
        apply_action_edits(
            [item], [{"Responsible": "", "Task": "Отчёт", "Deadline": None, "Status": "overdue"}]
        )
