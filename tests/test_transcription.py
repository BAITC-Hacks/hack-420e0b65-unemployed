import subprocess
from pathlib import Path

from minutes.models import Transcript
from minutes.transcription import transcribe


def test_native_gpu_crash_retries_cpu_and_reports_actual_device(tmp_path, monkeypatch):
    model = tmp_path / "whisper" / "tiny"
    model.mkdir(parents=True)
    (model / "model.bin").touch()
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
