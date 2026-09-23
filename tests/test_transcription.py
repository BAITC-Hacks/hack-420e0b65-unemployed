import subprocess
from pathlib import Path

from minutes.models import Transcript
from minutes.transcription import transcribe


def test_native_gpu_crash_retries_cpu_and_reports_actual_device(tmp_path, monkeypatch):
    model = tmp_path / "whisper" / "tiny"
    model.mkdir(parents=True)
    (model / "model.bin").touch()
    (model / "config.json").touch()
    (model / "tokenizer.json").touch()
    attempts = []

    def run(command, **kwargs):
        target = command[command.index("--device") + 1]
        attempts.append(target)
        if target == "cuda":
            return subprocess.CompletedProcess(command, -6, "", "missing libcudnn")
        output = Path(command[command.index("--output") + 1])
        output.write_text(
            Transcript(
                segments=[],
                language="ru",
                duration=1,
                device="cpu",
                model="tiny",
            ).model_dump_json()
        )
        return subprocess.CompletedProcess(command, 0, "", "")

    monkeypatch.setattr(subprocess, "run", run)
    result = transcribe(tmp_path / "audio.wav", tmp_path, "tiny")
    assert attempts == ["cuda", "cpu"]
    assert result.device == "cpu"
    assert "CPU" in result.warnings[0]


def test_mixed_passages_are_decoded_separately_with_original_timestamps():
    from types import SimpleNamespace

    import numpy as np

    from minutes.transcription import recognize_regions

    calls = []

    class Engine:
        def transcribe(self, audio, **kwargs):
            calls.append((len(audio), kwargs))
            text, language = ("Отчёт", "ru") if len(calls) == 1 else ("Есеп", "kk")
            word = SimpleNamespace(start=0.1, end=0.8, word=text)
            segment = SimpleNamespace(start=0.1, end=0.8, text=text, words=[word])
            return iter([segment]), SimpleNamespace(language=language)

    segments, languages = recognize_regions(
        Engine(),
        np.zeros(80000),
        [{"start": 0, "end": 16000}, {"start": 48000, "end": 64000}],
        None,
    )
    assert len(calls) == 2
    assert languages == ["ru", "kk"]
    assert [s.text for s in segments] == ["Отчёт", "Есеп"]
    assert segments[1].start == 3.1
    assert segments[1].words[0].start == 3.1


def test_partial_model_is_rejected_before_native_inference(tmp_path):
    import pytest

    model = tmp_path / "whisper" / "tiny"
    model.mkdir(parents=True)
    (model / "model.bin").touch()
    with pytest.raises(ValueError, match="incomplete"):
        transcribe(tmp_path / "audio.wav", tmp_path, "tiny")
