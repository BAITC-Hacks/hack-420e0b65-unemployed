import json
from datetime import date

import httpx
import pytest

from minutes.extraction import LocalOllama, chunks
from minutes.models import Segment, Transcript


def transcript():
    return Transcript(
        segments=[Segment(id=0, start=0, end=5, text="Я подготовлю отчёт завтра.")],
        language="ru",
        duration=5,
        device="cpu",
        model="test",
    )


def client_with_transport(monkeypatch, handler):
    llm = LocalOllama("http://localhost:11434", "local-model")
    monkeypatch.setattr(
        llm,
        "_client",
        lambda: httpx.Client(
            base_url=llm.url,
            transport=httpx.MockTransport(handler),
        ),
    )
    return llm


def test_schema_repair_returns_valid_result(monkeypatch):
    calls = []

    def handler(request):
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"model_info": {"architecture": "qwen3"}})
        calls.append(json.loads(request.content))
        raw = "not json" if len(calls) == 1 else '{"summary":"Обсудили отчёт", "action_items":[]}'
        return httpx.Response(200, json={"message": {"content": raw}})

    result = client_with_transport(monkeypatch, handler).extract(transcript(), date(2026, 9, 23))
    assert result.summary == "Обсудили отчёт"
    assert len(calls) == 2
    assert calls[0]["format"]["additionalProperties"] is False


def test_remote_model_rejected_before_meeting_is_sent(monkeypatch):
    def handler(request):
        assert request.url.path == "/api/show"
        assert "отчёт" not in request.content.decode()
        return httpx.Response(200, json={"remote_host": "https://ollama.com"})

    with pytest.raises(ValueError, match="remotely"):
        client_with_transport(monkeypatch, handler).extract(transcript(), date.today())


def test_failed_repair_is_bounded(monkeypatch):
    calls = []

    def handler(request):
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"model_info": {"architecture": "qwen3"}})
        calls.append(request)
        return httpx.Response(200, json={"message": {"content": "{}"}})

    with pytest.raises(ValueError, match="3 attempts"):
        client_with_transport(monkeypatch, handler).extract(transcript(), date.today())
    assert len(calls) == 3


def long_transcript():
    lines = [
        ("SPEAKER_00", "Ерлан, добавь резервную модель до конца дня."),
        ("SPEAKER_01", "Жарайды, мен бүгін кешке дейін резервтік модельді қосамын."),
        ("SPEAKER_02", "Мен сондай-ақ скриншоттарды дайындаймын, егер уақыт болса."),
        ("SPEAKER_00", "Да, презентацию я сделаю сам, отправлю всем в понедельник."),
    ] * 12  # 48 multi-speaker segments -> two extraction chunks.
    return Transcript(
        segments=[
            Segment(id=i, start=i * 5, end=i * 5 + 4, text=text + " " * 60, speaker=speaker)
            for i, (speaker, text) in enumerate(lines)
        ],
        language="ru+kk",
        duration=len(lines) * 5,
        device="cloud",
        model="gemini-test",
    )


def test_long_transcript_with_malformed_output_returns_verified_partial_minutes(monkeypatch):
    first_chunk = {
        "summary": "Распределили задачи.",
        "action_items": [
            {  # Valid: kept with its evidence.
                "task": "Добавить резервную модель",
                "responsible": "Ерлан",
                "responsible_speaker": None,
                "deadline": None,
                "deadline_text": "до конца дня",
                "evidence_segment_ids": [0],
                "evidence_quote": "Ерлан, добавь резервную модель до конца дня.",
                "status": "in progress",
            },
            {  # Paraphrased quote ("скриншоты"): dropped, must not sink the others.
                "task": "Подготовить скриншоты",
                "responsible": None,
                "responsible_speaker": "SPEAKER_02",
                "deadline": None,
                "deadline_text": None,
                "evidence_segment_ids": [2],
                "evidence_quote": "Мен сондай-ақ скриншоты дайындаймын",
                "status": "in progress",
            },
            {  # Schema-invalid deadline: dropped instead of failing the whole chunk.
                "task": "Что-то",
                "responsible": None,
                "responsible_speaker": None,
                "deadline": "next Friday",
                "deadline_text": None,
                "evidence_segment_ids": [1],
                "evidence_quote": "резервтік модельді қосамын",
                "status": "in progress",
            },
            {  # "сам" is a pronoun, not a person: responsible must become null.
                "task": "Сделать презентацию",
                "responsible": "Сам",
                "responsible_speaker": None,
                "deadline": None,
                "deadline_text": "в понедельник",
                "evidence_segment_ids": [3],
                "evidence_quote": "презентацию я сделаю сам, отправлю всем в понедельник",
                "status": "in progress",
            },
        ],
    }
    chats = []

    def handler(request):
        if request.url.path == "/api/show":
            return httpx.Response(200, json={"model_info": {"architecture": "qwen3"}})
        chats.append(request)
        if len(chats) == 1:
            raw = json.dumps(first_chunk, ensure_ascii=False)
        else:  # Second chunk: truncated JSON on every attempt.
            raw = '{"summary": "Вторая часть", "action_items": [{"task": "обре'
        return httpx.Response(200, json={"message": {"content": raw}})

    llm = client_with_transport(monkeypatch, handler)
    transcript = long_transcript()
    assert len(chunks(transcript.segments)) == 2
    result = llm.extract(transcript, date(2026, 9, 23))
    assert len(chats) == 4  # 1 for chunk one, 3 bounded attempts for chunk two.
    assert [(a.task, a.responsible, a.evidence_segment_ids) for a in result.action_items] == [
        ("Добавить резервную модель", "Ерлан", [0]),
        ("Сделать презентацию", None, [3]),
    ]
    assert result.action_items[0].deadline is None  # "до конца дня" is not resolved/guessed.
    assert result.action_items[1].deadline == date(2026, 9, 28)
    assert result.summary.startswith("Распределили задачи.")
    assert "Не обработано автоматически" in result.summary
