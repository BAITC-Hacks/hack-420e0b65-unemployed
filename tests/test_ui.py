from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_starts_without_models_or_audio():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=20)
    assert not app.exception
    assert app.title[0].value == "Alem Minutes"
    assert any(button.label == "1. Transcribe locally" and button.disabled for button in app.button)


def test_mapping_updates_transcript_and_invalidates_old_minutes():
    from datetime import date

    from minutes.models import Extraction, Meeting, Segment, Transcript

    meeting = Meeting(
        title="Тест",
        meeting_date=date(2026, 9, 23),
        transcript=Transcript(
            segments=[Segment(id=0, start=0, end=2, text="Я сделаю отчёт.", speaker="SPEAKER_00")],
            language="ru",
            duration=2,
            device="cpu",
            model="test fixture",
        ),
        extraction=Extraction(summary="Old anonymous summary", action_items=[]),
    )
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py")
    app.session_state["meeting"] = meeting
    app.run(timeout=20)
    next(field for field in app.text_input if field.label.startswith("SPEAKER_00")).set_value(
        "Айгүл"
    )
    next(button for button in app.button if button.label == "Apply participant names").click().run()
    assert not app.exception
    updated = app.session_state["meeting"]
    assert updated.speaker_names == {"SPEAKER_00": "Айгүл"}
    assert updated.extraction is None
    assert "Айгүл" in app.text_area[0].value


def test_action_dashboard_and_exports_render_with_unspecified_deadline():
    from datetime import date

    from minutes.models import ActionItem, Extraction, Meeting, Segment, Transcript

    action = ActionItem(
        task="Проверить отчёт",
        responsible=None,
        responsible_speaker=None,
        deadline=None,
        deadline_text="когда-нибудь потом",
        evidence_segment_ids=[0],
        evidence_quote="Проверить отчёт",
    )
    meeting = Meeting(
        title="Тест",
        meeting_date=date(2026, 9, 23),
        transcript=Transcript(
            segments=[Segment(id=0, start=0, end=2, text="Проверить отчёт")],
            language="ru",
            duration=2,
            device="cpu",
            model="test fixture",
        ),
        extraction=Extraction(summary="Проверка отчёта", action_items=[action]),
    )
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py")
    app.session_state["meeting"] = meeting
    app.run(timeout=20)
    assert not app.exception
    assert len(app.metric) == 3
    assert list(app.dataframe[0].value["Deadline"]) == ["—"]
    assert len(app.get("download_button")) == 4
    next(button for button in app.button if button.label == "Save action updates").click().run()
    assert not app.exception
    assert app.session_state["meeting"].extraction.action_items[0].deadline is None
