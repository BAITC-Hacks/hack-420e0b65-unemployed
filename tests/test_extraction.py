import json
from datetime import date

import httpx
import pytest

from minutes.extraction import LocalOllama
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
