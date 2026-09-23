"""Failure-mode coverage for local model output and operator configuration mistakes."""

import pytest

from minutes.config import Settings
from minutes.diarization import assign_speakers
from minutes.extraction import json_object, normalize, parse_extraction, validate_evidence
from minutes.models import Extraction, Segment, SpeakerTurn, Transcript, Word

PAYLOAD = '{"summary": "Обсудили отчёт", "action_items": []}'


@pytest.mark.parametrize(
    "raw",
    [
        PAYLOAD,
        f"```json\n{PAYLOAD}\n```",
        f"Вот результат:\n{PAYLOAD}\nГотово.",
        f"<think>The speaker promises a report.</think>\n{PAYLOAD}",
        f'Explanation with a }} brace and "quoted {{" text.\n{PAYLOAD}',
    ],
)
def test_minutes_survive_wrapped_or_prefixed_model_output(raw):
    assert parse_extraction(raw).summary == "Обсудили отчёт"


def test_nested_objects_are_returned_whole():
    text = 'noise {"a": {"b": [1, 2]}, "c": "}"} tail'
    assert json_object(text) == '{"a": {"b": [1, 2]}, "c": "}"}'


def test_output_without_json_is_rejected_not_guessed():
    with pytest.raises(ValueError):
        parse_extraction("The model refused to answer.")


def test_quote_whitespace_differences_do_not_discard_a_real_action():
    segments = [Segment(id=0, start=0, end=5, text="Я подготовлю отчёт к пятнице.")]
    value = Extraction.model_validate(
        {
            "summary": "Отчёт",
            "action_items": [
                {
                    "task": "Подготовить отчёт",
                    "responsible": "Айгүл",
                    "responsible_speaker": None,
                    "deadline": None,
                    "deadline_text": "к пятнице",
                    "evidence_segment_ids": [0],
                    "evidence_quote": "Я  подготовлю\nотчёт",
                }
            ],
        }
    )
    assert validate_evidence(value, segments, {}).action_items[0].responsible == "Айгүл"


def test_invented_quote_is_still_rejected():
    segments = [Segment(id=0, start=0, end=5, text="Обсудили бюджет.")]
    value = Extraction.model_validate(
        {
            "summary": "Бюджет",
            "action_items": [
                {
                    "task": "Купить сервер",
                    "responsible": None,
                    "responsible_speaker": None,
                    "deadline": None,
                    "deadline_text": None,
                    "evidence_segment_ids": [0],
                    "evidence_quote": "Купим сервер завтра",
                }
            ],
        }
    )
    with pytest.raises(ValueError, match="verbatim"):
        validate_evidence(value, segments, {})


def test_normalize_collapses_whitespace():
    assert normalize("  а\n б\tв ") == "а б в"


def test_invalid_device_in_env_is_reported_not_crashed(monkeypatch):
    monkeypatch.setenv("WHISPER_DEVICE", "gpu")
    with pytest.raises(ValueError, match="WHISPER_DEVICE"):
        Settings.load()


def test_defaults_load_without_any_env_file(monkeypatch):
    for name in ("WHISPER_DEVICE", "WHISPER_MODEL", "OLLAMA_URL", "OLLAMA_MODEL", "MODEL_DIR"):
        monkeypatch.delenv(name, raising=False)
    settings = Settings.load()
    assert settings.whisper_device == "auto"
    assert settings.ollama_url == "http://127.0.0.1:11434"


def test_blank_word_tokens_do_not_abort_speaker_assignment():
    source = Transcript(
        segments=[
            Segment(
                id=0,
                start=0,
                end=4,
                text="Да",
                words=[Word(start=0, end=1, text=" "), Word(start=3, end=4, text="Да")],
            )
        ],
        language="ru",
        duration=4,
        device="cpu",
        model="test",
    )
    turns = [
        SpeakerTurn(start=0, end=1.5, speaker="SPEAKER_00"),
        SpeakerTurn(start=2.5, end=4, speaker="SPEAKER_01"),
    ]
    result = assign_speakers(source, turns)
    assert [(s.id, s.text, s.speaker) for s in result.segments] == [(0, "Да", "SPEAKER_01")]


def test_undecodable_upload_reports_a_user_error_not_a_traceback(tmp_path, monkeypatch):
    import subprocess

    from minutes.transcription import DECODE_ERROR, transcribe

    model = tmp_path / "whisper" / "tiny"
    model.mkdir(parents=True)
    for name in ("model.bin", "config.json", "tokenizer.json"):
        (model / name).touch()
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda command, **kwargs: subprocess.CompletedProcess(
            command, 1, "", f"{DECODE_ERROR}: InvalidDataError: bad header"
        ),
    )
    with pytest.raises(ValueError, match="could not be decoded"):
        transcribe(tmp_path / "audio.wav", tmp_path, "tiny")
