import json
from datetime import date

import pytest
from pydantic import ValidationError

from minutes.config import local_ollama_url
from minutes.extraction import chunks, parse_extraction, validate_evidence
from minutes.models import ActionItem, Extraction, Segment


def action(**overrides):
    data = dict(
        task="Подготовить отчёт",
        responsible=None,
        responsible_speaker=None,
        deadline="2026-09-24",
        deadline_text="завтра",
        evidence_segment_ids=[0],
        evidence_quote="Я подготовлю отчёт завтра.",
    )
    data.update(overrides)
    return ActionItem(**data)


@pytest.mark.parametrize(
    "url",
    [
        "https://api.openai.com",
        "http://192.168.1.1:11434",
        "http://localhost.evil.test",
        "http://user@localhost:11434",
        "http://localhost:11434/path",
        "http://localhost:11434?remote=1",
    ],
)
def test_external_endpoints_rejected(url):
    with pytest.raises(ValueError):
        local_ollama_url(url)


def test_loopback_is_canonicalized():
    assert local_ollama_url("http://localhost:11434/") == "http://127.0.0.1:11434"


def test_json_and_fenced_json_validate():
    raw = json.dumps(
        dict(summary="Задача согласована", action_items=[action().model_dump(mode="json")])
    )
    assert len(parse_extraction(raw).action_items) == 1
    assert parse_extraction(f"```json\n{raw}\n```").summary == "Задача согласована"


def test_malformed_and_extra_fields_rejected():
    with pytest.raises(ValidationError):
        parse_extraction('{"summary":"ok","action_items":[],"made_up":true}')
    with pytest.raises(ValidationError):
        action(deadline="next Friday")


def test_evidence_and_speaker_mapping():
    segment = Segment(id=0, start=0, end=4, text="Я подготовлю отчёт завтра.", speaker="SPEAKER_00")
    result = Extraction(summary="Отчёт", action_items=[action(responsible_speaker="SPEAKER_00")])
    assert (
        validate_evidence(result, [segment], {"SPEAKER_00": "Айгүл"}).action_items[0].responsible
        == "Айгүл"
    )
    result.action_items[0].evidence_quote = "Invented quote"
    with pytest.raises(ValueError, match="verbatim"):
        validate_evidence(result, [segment], {})


def test_unknown_source_is_rejected():
    result = Extraction(summary="Отчёт", action_items=[action(evidence_segment_ids=[5])])
    with pytest.raises(ValueError, match="unknown segment"):
        validate_evidence(result, [], {})


def test_chunking_preserves_every_segment():
    segments = [Segment(id=i, start=i, end=i + 1, text="Жиналыс " * 20) for i in range(30)]
    groups = chunks(segments, limit=700)
    assert len(groups) > 1
    assert [s for group in groups for s in group] == segments


def test_overdue_is_derived_and_completed_wins():
    item = action()
    assert item.display_status(date(2026, 9, 24)) == "in progress"
    assert item.display_status(date(2026, 9, 25)) == "overdue"
    item.status = "completed"
    assert item.display_status(date(2026, 9, 25)) == "completed"


def test_bad_segment_times_rejected():
    with pytest.raises(ValidationError):
        Segment(id=0, start=5, end=1, text="text")
