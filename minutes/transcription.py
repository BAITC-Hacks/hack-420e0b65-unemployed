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

DECODE_ERROR = "AUDIO_DECODE_ERROR"
TARGET_LANGUAGES = ("ru", "kk")


def transcribe(
    audio: Path,
    model_dir: Path,
    model: str = "large-v3",
    device: str = "auto",
    language: str | None = None,
) -> Transcript:
    model_path = model_dir / "whisper" / model
    if not all(
        (model_path / name).is_file() for name in ("model.bin", "config.json", "tokenizer.json")
    ):
        raise ValueError(
            f"Whisper model missing or incomplete: {model_path}. Run scripts/download_models.py."
        )
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
            if DECODE_ERROR in result.stderr:
                raise ValueError(
                    "This file could not be decoded as audio. Upload a valid WAV, MP3, M4A, "
                    "OGG, FLAC or MP4 recording."
                )
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


def decode(engine, audio, language: str | None):
    return engine.transcribe(
        audio,
        language=language,
        beam_size=5,
        vad_filter=False,
        word_timestamps=True,
        condition_on_previous_text=False,
        multilingual=language is None,
    )


def recognize_regions(engine, samples, regions: list[dict], language: str | None):
    rows, languages = [], []
    for region in regions:
        offset = region["start"] / 16000
        audio = samples[region["start"] : region["end"]]
        segments, info = decode(engine, audio, language)
        if language is None and info.language not in TARGET_LANGUAGES:
            # Short passages are sometimes detected as an unrelated language, which
            # produces transliterated nonsense. This product only handles RU/KZ.
            fallback = next(
                (code for code in languages if code in TARGET_LANGUAGES),
                "ru",
            )
            segments, info = decode(engine, audio, fallback)
        if info.language not in languages:
            languages.append(info.language)
        for segment in segments:
            if segment.text.strip():
                rows.append(
                    Segment(
                        id=len(rows),
                        start=segment.start + offset,
                        end=segment.end + offset,
                        text=segment.text,
                        words=[
                            Word(start=word.start + offset, end=word.end + offset, text=word.word)
                            for word in (segment.words or [])
                        ],
                    )
                )
    return rows, languages


def worker(audio: str, model: str, device: str, language: str | None) -> Transcript:
    import onnxruntime
    from faster_whisper import WhisperModel
    from faster_whisper.audio import decode_audio
    from faster_whisper.vad import VadOptions, get_speech_timestamps

    onnxruntime.disable_telemetry_events()
    try:
        samples = decode_audio(audio, sampling_rate=16000)
    except Exception as exc:  # PyAV raises container-specific errors for unreadable media.
        raise SystemExit(f"{DECODE_ERROR}: {type(exc).__name__}: {exc}") from exc
    duration = len(samples) / 16000
    if duration > 7200:
        raise ValueError("Recording exceeds the two-hour limit; split it into shorter meetings")
    # Decode separate voiced passages: concatenating them into one 30-second window
    # caused a Russian passage to suppress a following Kazakh passage in real testing.
    regions = get_speech_timestamps(
        samples,
        VadOptions(
            min_silence_duration_ms=350,
            speech_pad_ms=200,
            max_speech_duration_s=25,
        ),
    )
    if not regions:
        return Transcript(
            segments=[], language=language or "und", duration=duration, device=device, model=model
        )
    engine = WhisperModel(
        model,
        device=device,
        compute_type="int8_float16" if device == "cuda" else "int8",
        local_files_only=True,
        cpu_threads=min(8, os.cpu_count() or 4),
    )
    rows, languages = recognize_regions(engine, samples, regions, language)
    del engine
    gc.collect()
    return Transcript(
        segments=rows,
        language="+".join(languages),
        duration=duration,
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
