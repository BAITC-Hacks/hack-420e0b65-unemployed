"""Real offline ONNX diarization and timestamp-overlap transcript attribution."""

import argparse
import json
import subprocess
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

from minutes.models import Segment, SpeakerTurn, Transcript


def best_speaker(start: float, end: float, turns: list[SpeakerTurn]) -> str | None:
    overlaps: dict[str, float] = defaultdict(float)
    for turn in turns:
        overlap = max(0.0, min(end, turn.end) - max(start, turn.start))
        if overlap:
            overlaps[turn.speaker] += overlap
    ranked = sorted(overlaps.items(), key=lambda item: item[1], reverse=True)
    if not ranked:
        return nearest_speaker(start, end, turns)
    if len(ranked) > 1 and abs(ranked[0][1] - ranked[1][1]) < 0.001:
        return None  # Simultaneous voices cannot be resolved reliably by timestamp overlap.
    return ranked[0][0]


def nearest_speaker(
    start: float, end: float, turns: list[SpeakerTurn], tolerance: float = 0.3
) -> str | None:
    """Words in a short pause between turns belong to the adjacent turn, if unambiguous."""
    gaps: dict[str, float] = {}
    for turn in turns:
        gap = max(turn.start - end, start - turn.end)
        if gap <= tolerance:
            gaps[turn.speaker] = min(gap, gaps.get(turn.speaker, gap))
    ranked = sorted(gaps.items(), key=lambda item: item[1])
    if not ranked or (len(ranked) > 1 and abs(ranked[0][1] - ranked[1][1]) < 0.05):
        return None  # Equidistant from two speakers: keep unknown instead of guessing.
    return ranked[0][0]


def assign_speakers(transcript: Transcript, turns: list[SpeakerTurn]) -> Transcript:
    rows = []
    for segment in transcript.segments:
        if not segment.words:
            rows.append(
                segment.model_copy(
                    update={
                        "speaker": best_speaker(segment.start, segment.end, turns),
                        "id": len(rows),
                    }
                )
            )
            continue
        groups = []
        for word in segment.words:
            speaker = best_speaker(word.start, word.end, turns)
            if groups and groups[-1][0] == speaker:
                groups[-1][1].append(word)
            else:
                groups.append((speaker, [word]))
        for speaker, words in groups:
            text = " ".join(w.text for w in words if w.text).strip()
            if not text:
                continue  # Whitespace-only word tokens must not abort the whole meeting.
            rows.append(
                Segment(
                    id=len(rows),
                    start=words[0].start,
                    end=words[-1].end,
                    text=text,
                    speaker=speaker,
                    words=words,
                )
            )
    return transcript.model_copy(update={"segments": rows})


def diarize(audio: Path, model_dir: Path, num_speakers: int = 0) -> list[SpeakerTurn]:
    if not 0 <= num_speakers <= 20:
        raise ValueError("Number of speakers must be 0 (automatic) to 20")
    folder = model_dir / "diarization"
    for name in ("segmentation.onnx", "embedding.onnx"):
        if not (folder / name).is_file():
            raise ValueError(
                "Diarization models missing. Run scripts/download_models.py --diarization"
            )
    with tempfile.TemporaryDirectory(prefix="alem-speakers-") as work:
        output = Path(work) / "turns.json"
        try:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "minutes.diarization",
                    "--audio",
                    str(audio.resolve()),
                    "--model-dir",
                    str(folder.resolve()),
                    "--speakers",
                    str(num_speakers),
                    "--output",
                    str(output),
                ],
                capture_output=True,
                text=True,
                timeout=3600,
                check=False,
                cwd=Path(__file__).resolve().parent.parent,
            )
        except subprocess.TimeoutExpired as exc:
            raise RuntimeError("Diarization exceeded the one-hour processing limit") from exc
        if result.returncode != 0:
            raise RuntimeError(f"Local diarization failed: {result.stderr[-1000:]}")
        return [SpeakerTurn.model_validate(row) for row in json.loads(output.read_text())]


def worker(audio: str, folder: Path, speakers: int) -> list[SpeakerTurn]:
    import sherpa_onnx
    from faster_whisper.audio import decode_audio

    config = sherpa_onnx.OfflineSpeakerDiarizationConfig(
        segmentation=sherpa_onnx.OfflineSpeakerSegmentationModelConfig(
            pyannote=sherpa_onnx.OfflineSpeakerSegmentationPyannoteModelConfig(
                model=str(folder / "segmentation.onnx"),
            ),
            num_threads=4,
            provider="cpu",
        ),
        embedding=sherpa_onnx.SpeakerEmbeddingExtractorConfig(
            model=str(folder / "embedding.onnx"),
            num_threads=4,
            provider="cpu",
        ),
        clustering=sherpa_onnx.FastClusteringConfig(
            num_clusters=speakers if speakers else -1,
            threshold=0.5,
        ),
        min_duration_on=0.3,
        min_duration_off=0.5,
    )
    if not config.validate():
        raise ValueError("Invalid diarization model configuration")
    engine = sherpa_onnx.OfflineSpeakerDiarization(config)
    samples = decode_audio(audio, sampling_rate=engine.sample_rate)
    if len(samples) < engine.sample_rate // 2:
        return []
    result = engine.process(samples).sort_by_start_time()
    labels = {}
    rows = []
    for turn in result:
        if turn.speaker not in labels:
            labels[turn.speaker] = f"SPEAKER_{len(labels):02}"
        rows.append(SpeakerTurn(start=turn.start, end=turn.end, speaker=labels[turn.speaker]))
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--model-dir", required=True, type=Path)
    parser.add_argument("--speakers", type=int, default=0)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    turns = worker(args.audio, args.model_dir, args.speakers)
    args.output.write_text(json.dumps([turn.model_dump() for turn in turns]))
