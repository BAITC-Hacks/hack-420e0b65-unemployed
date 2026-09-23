"""Isolate native inference: release VRAM and survive CUDA library failures."""

import argparse
import gc
import json
import os
import subprocess
import sys
import sysconfig
import tempfile
from pathlib import Path

from minutes.models import Segment, Transcript, Word


def transcribe(
    audio: Path,
    model_dir: Path,
    model: str = "large-v3",
    device: str = "auto",
    language: str | None = None,
) -> Transcript:
    model_path = model_dir / "whisper" / model
    if not (model_path / "model.bin").is_file():
        raise ValueError(f"Whisper model missing: {model_path}. Run scripts/download_models.py.")
    if device not in {"auto", "cuda", "cpu"}:
        raise ValueError("Device must be auto, cuda or cpu")
    devices = ["cuda", "cpu"] if device in {"auto", "cuda"} else ["cpu"]
    warnings = []
    env = os.environ.copy()
    env.update(HF_HUB_OFFLINE="1", HF_HUB_DISABLE_TELEMETRY="1")
    libs = Path(sysconfig.get_paths()["purelib"]) / "nvidia"
    library_paths = [str(libs / p / "lib") for p in ("cublas", "cudnn")]
    env["LD_LIBRARY_PATH"] = ":".join(library_paths + [env.get("LD_LIBRARY_PATH", "")])
    with tempfile.TemporaryDirectory(prefix="alem-stt-") as work:
        result_path = Path(work) / "transcript.json"
        for target in devices:
            command = [
                sys.executable,
                "-m",
                "minutes.transcription",
                "--audio",
                str(audio.resolve()),
                "--model",
                str(model_path.resolve()),
                "--device",
                target,
                "--output",
                str(result_path),
            ]
            if language:
                command += ["--language", language]
            try:
                result = subprocess.run(
                    command,
                    capture_output=True,
                    text=True,
                    env=env,
                    timeout=7200,
                    cwd=Path(__file__).resolve().parent.parent,
                )
            except subprocess.TimeoutExpired as exc:
                raise RuntimeError("Transcription exceeded the 2-hour processing limit.") from exc
            if result.returncode == 0 and result_path.exists():
                transcript = Transcript.model_validate_json(result_path.read_text())
                transcript.model = model
                transcript.warnings.extend(warnings)
                return transcript
            if target == "cuda":
                warnings.append("CUDA transcription failed; automatically retried on CPU (int8).")
            else:
                # Decoder/model failures do not contain transcript data; only keep a short tail.
                raise RuntimeError(f"Local transcription failed: {result.stderr[-1500:]}")
    raise RuntimeError("No transcription backend available")


def worker(audio: str, model: str, device: str, language: str | None) -> Transcript:
    from faster_whisper import WhisperModel

    engine = WhisperModel(
        model,
        device=device,
        compute_type="int8_float16" if device == "cuda" else "int8",
        local_files_only=True,
        cpu_threads=min(8, os.cpu_count() or 4),
    )
    segments, info = engine.transcribe(
        audio,
        language=language,
        beam_size=5,
        vad_filter=True,
        word_timestamps=True,
        condition_on_previous_text=False,
        multilingual=language is None,
    )
    rows = []
    for s in segments:
        if s.text.strip():
            rows.append(
                Segment(
                    id=len(rows),
                    start=s.start,
                    end=s.end,
                    text=s.text,
                    words=[Word(start=w.start, end=w.end, text=w.word) for w in (s.words or [])],
                )
            )
    del engine
    gc.collect()
    return Transcript(
        segments=rows,
        language=info.language,
        duration=info.duration,
        device=device,
        model=model,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--language", default=None)
    args = parser.parse_args()
    output = worker(args.audio, args.model, args.device, args.language)
    Path(args.output).write_text(json.dumps(output.model_dump(mode="json"), ensure_ascii=False))
