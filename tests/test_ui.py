from pathlib import Path

from streamlit.testing.v1 import AppTest


def test_app_starts_without_models_or_audio():
    app = AppTest.from_file(Path(__file__).resolve().parents[1] / "app.py").run(timeout=20)
    assert not app.exception
    assert app.title[0].value == "Alem Minutes"
    assert any(button.label == "1. Transcribe locally" and button.disabled for button in app.button)
