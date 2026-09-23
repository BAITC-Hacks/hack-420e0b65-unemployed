import json
from pathlib import Path

import httpx
import pytest

from minutes import gemini

FAKE_KEY = "test-key-not-real-123"


def response_body(payload, finish="STOP"):
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return {"candidates": [{"content": {"parts": [{"text": text}]}, "finishReason": finish}]}


SEGMENTS = {
    "language": "ru+kk",
    "segments": [
        {"speaker": "Speaker 1", "start": 0, "end": 3.5, "text": "Коллеги, начинаем."},
        {"speaker": "Speaker 2", "start": 3.5, "end": 8, "text": "Жарайды, мен дайындаймын."},
        {"speaker": "speaker 1", "start": 8, "end": 9, "text": "Отлично."},
    ],
}


def test_parse_maps_speakers_and_keeps_coherent_timestamps():
    t = gemini.parse_response(response_body(SEGMENTS), duration=9.2, model="m")
    assert [(s.id, s.speaker, s.start, s.end) for s in t.segments] == [
        (0, "SPEAKER_00", 0, 3.5),
        (1, "SPEAKER_01", 3.5, 8),
        (2, "SPEAKER_00", 8, 9),
    ]
    assert t.segments[1].text == "Жарайды, мен дайындаймын."
    assert t.device.startswith("cloud")
    assert "Gemini" in t.warnings[0]


def test_incoherent_timestamps_are_not_invented():
    bad = {"segments": [{"speaker": "A", "start": 50, "end": 40, "text": "Текст"}]}
    t = gemini.parse_response(response_body(bad), duration=10, model="m")
    assert (t.segments[0].start, t.segments[0].end) == (0, 0)
    assert any("omitted" in w for w in t.warnings)


@pytest.mark.parametrize(
    ("body", "match"),
    [
        (response_body(SEGMENTS, finish="MAX_TOKENS"), "truncated"),
        ({"promptFeedback": {"blockReason": "SAFETY"}}, "SAFETY"),
        (response_body("not json at all"), "unreadable"),
    ],
)
def test_bad_responses_raise_clean_errors(body, match):
    with pytest.raises(RuntimeError, match=match):
        gemini.parse_response(body, duration=5, model="m")


def test_missing_key_fails_before_any_request(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setattr(gemini, "to_wav", lambda audio: pytest.fail("audio must not be read"))
    with pytest.raises(ValueError, match="GEMINI_API_KEY"):
        gemini.transcribe_gemini(Path("x.wav"))


def mocked(monkeypatch, handler):
    monkeypatch.setenv("GEMINI_API_KEY", FAKE_KEY)
    monkeypatch.setattr(gemini, "to_wav", lambda audio: (b"RIFFfake", 9.2))
    monkeypatch.setattr(gemini, "RETRY_DELAY", 0)
    real_client = httpx.Client
    monkeypatch.setattr(
        gemini.httpx,
        "Client",
        lambda **kwargs: real_client(transport=httpx.MockTransport(handler), **kwargs),
    )


def test_request_sends_key_only_in_header_and_retries_overload(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request)
        assert request.headers["x-goog-api-key"] == FAKE_KEY
        assert FAKE_KEY not in str(request.url)
        if len(calls) == 1:
            return httpx.Response(503, json={"error": {"message": "high demand"}})
        body = json.loads(request.content)
        assert body["contents"][0]["parts"][1]["inline_data"]["mime_type"] == "audio/wav"
        assert "VERBATIM" in body["contents"][0]["parts"][0]["text"]
        return httpx.Response(200, json=response_body(SEGMENTS))

    mocked(monkeypatch, handler)
    t = gemini.transcribe_gemini(Path("x.wav"), "gemini-test")
    assert len(calls) == 2
    assert calls[1].url.path == "/v1beta/models/gemini-test:generateContent"
    assert len(t.segments) == 3


def test_api_error_is_clean_and_never_contains_key(monkeypatch):
    mocked(
        monkeypatch,
        lambda request: httpx.Response(400, json={"error": {"message": "API key not valid"}}),
    )
    with pytest.raises(RuntimeError, match="Gemini API error 400") as info:
        gemini.transcribe_gemini(Path("x.wav"))
    assert FAKE_KEY not in str(info.value)


def test_network_failure_is_wrapped(monkeypatch):
    def handler(request):
        raise httpx.ConnectError("offline")

    mocked(monkeypatch, handler)
    with pytest.raises(RuntimeError, match="unreachable"):
        gemini.transcribe_gemini(Path("x.wav"))


def overloaded_primary(calls):
    def handler(request):
        calls.append(request.url.path)
        if gemini.DEFAULT_MODEL in request.url.path:
            return httpx.Response(503, json={"error": {"message": "high demand"}})
        return httpx.Response(200, json=response_body(SEGMENTS))

    return handler


def test_default_model_503_falls_back_once_and_reports_model(monkeypatch):
    calls = []
    mocked(monkeypatch, overloaded_primary(calls))
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    t = gemini.transcribe_gemini(Path("x.wav"))
    assert calls == [f"/v1beta/models/{gemini.DEFAULT_MODEL}:generateContent"] * 3 + [
        f"/v1beta/models/{gemini.FALLBACK_MODEL}:generateContent"
    ]
    assert t.model == gemini.FALLBACK_MODEL
    assert any("fallback" in w and gemini.FALLBACK_MODEL in w for w in t.warnings)


def test_explicit_model_choice_is_not_overridden(monkeypatch):
    calls = []
    mocked(monkeypatch, overloaded_primary(calls))
    monkeypatch.setenv("GEMINI_MODEL", gemini.DEFAULT_MODEL)
    with pytest.raises(RuntimeError, match="503"):
        gemini.transcribe_gemini(Path("x.wav"))
    assert all(gemini.FALLBACK_MODEL not in path for path in calls)


@pytest.mark.parametrize("status", [400, 401, 403, 429])
def test_non_overload_errors_never_fall_back(monkeypatch, status):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(status, json={"error": {"message": "nope"}})

    mocked(monkeypatch, handler)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    with pytest.raises(RuntimeError, match=str(status)):
        gemini.transcribe_gemini(Path("x.wav"))
    assert all(gemini.FALLBACK_MODEL not in path for path in calls)


def test_blocked_primary_response_does_not_fall_back(monkeypatch):
    calls = []

    def handler(request):
        calls.append(request.url.path)
        return httpx.Response(200, json={"promptFeedback": {"blockReason": "SAFETY"}})

    mocked(monkeypatch, handler)
    monkeypatch.delenv("GEMINI_MODEL", raising=False)
    with pytest.raises(RuntimeError, match="SAFETY"):
        gemini.transcribe_gemini(Path("x.wav"))
    assert len(calls) == 1
