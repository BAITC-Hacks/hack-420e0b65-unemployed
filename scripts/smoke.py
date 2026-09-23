"""Real local inference smoke. No model download and no synthetic AI responses."""

import argparse
import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minutes.config import Settings
from minutes.diarization import assign_speakers, diarize
from minutes.exports import export_docx, export_pdf
from minutes.extraction import LocalOllama
from minutes.models import Meeting
from minutes.transcription import transcribe


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("audio", type=Path)
    parser.add_argument("--model", default="tiny")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto")
    parser.add_argument("--language", choices=["ru", "kk", "en"])
    parser.add_argument("--diarize", action="store_true")
    parser.add_argument("--speakers", type=int, default=0)
    parser.add_argument("--output", type=Path, default=Path("artifacts/smoke.json"))
    args = parser.parse_args()
    settings = Settings.load()
    transcript = transcribe(args.audio, settings.model_dir, args.model, args.device, args.language)
    print(
        f"STT: {len(transcript.segments)} segments; device={transcript.device}; language={transcript.language}"
    )
    for warning in transcript.warnings:
        print(warning)
    if not transcript.segments:
        raise SystemExit("No speech recognized; smoke failed")
    turns = []
    if args.diarize:
        turns = diarize(args.audio, settings.model_dir, args.speakers)
        if not turns:
            raise SystemExit("No speaker turns detected; smoke failed")
        transcript = assign_speakers(transcript, turns)
        print(f"Diarization: {len(turns)} turns, {len({t.speaker for t in turns})} speakers")
    result = LocalOllama(settings.ollama_url, settings.ollama_model).extract(
        transcript, date(2026, 9, 23)
    )
    meeting = Meeting(
        title="Local inference verification",
        meeting_date=date(2026, 9, 23),
        transcript=transcript,
        extraction=result,
        speaker_turns=turns,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(meeting.model_dump_json(indent=2), encoding="utf-8")
    args.output.with_suffix(".docx").write_bytes(export_docx(meeting))
    args.output.with_suffix(".pdf").write_bytes(export_pdf(meeting))
    print(f"LLM: validated summary and {len(result.action_items)} actions; saved to {args.output}")


if __name__ == "__main__":
    main()
