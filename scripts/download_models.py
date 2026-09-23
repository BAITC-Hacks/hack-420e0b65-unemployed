"""Explicit, immutable model download. Standard library only; no runtime downloads."""

import argparse
import hashlib
import json
import tarfile
import time
from pathlib import Path
from urllib.request import Request, urlopen


def download(url: str, destination: Path, sha256: str | None = None):
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and (not sha256 or checksum(destination) == sha256):
        return
    partial = destination.with_suffix(destination.suffix + ".part")
    for attempt in range(3):
        try:
            offset = partial.stat().st_size if partial.exists() else 0
            request = Request(url, headers={"Range": f"bytes={offset}-"} if offset else {})
            with urlopen(request, timeout=90) as response:
                mode = "ab" if offset and response.status == 206 else "wb"
                with partial.open(mode) as output:
                    while block := response.read(1024 * 1024):
                        output.write(block)
            if sha256 and checksum(partial) != sha256:
                partial.unlink()
                raise ValueError(f"Checksum mismatch: {destination.name}")
            partial.replace(destination)
            print(f"Ready: {destination}", flush=True)
            return
        except (OSError, ValueError):
            if attempt == 2:
                raise
            time.sleep(2)


def checksum(path: Path) -> str:
    with path.open("rb") as source:
        return hashlib.file_digest(source, "sha256").hexdigest()


def whisper(name: str, root: Path):
    sources = json.loads(Path(__file__).with_name("model_sources.json").read_text())
    if name not in sources:
        raise ValueError(f"Unpinned model {name!r}; supported: {', '.join(sources)}")
    entry = sources[name]
    destination = root / "whisper" / name
    for item in entry["files"]:
        filename = item["rfilename"]
        url = f"https://huggingface.co/{entry['repo']}/resolve/{entry['revision']}/{filename}"
        download(url, destination / filename, item.get("lfs", {}).get("sha256"))
    (destination / "source.json").write_text(json.dumps(entry, indent=2))
    print(f"Whisper ready: {destination}", flush=True)


def diarization(root: Path):
    folder = root / "diarization"
    base = "https://github.com/k2-fsa/sherpa-onnx/releases/download/"
    archive = folder / "segmentation.tar.bz2"
    download(
        base + "speaker-segmentation-models/sherpa-onnx-pyannote-segmentation-3-0.tar.bz2",
        archive,
        "24615ee884c897d9d2ba09bb4d30da6bb1b15e685065962db5b02e76e4996488",
    )
    with tarfile.open(archive, "r:bz2") as bundle:
        member = bundle.getmember("sherpa-onnx-pyannote-segmentation-3-0/model.onnx")
        with bundle.extractfile(member) as source:
            (folder / "segmentation.onnx").write_bytes(source.read())
        with bundle.extractfile("sherpa-onnx-pyannote-segmentation-3-0/LICENSE") as source:
            (folder / "SEGMENTATION-LICENSE").write_bytes(source.read())
    download(
        base
        + "speaker-recongition-models/3dspeaker_speech_eres2net_base_sv_zh-cn_3dspeaker_16k.onnx",
        folder / "embedding.onnx",
        "1a331345f04805badbb495c775a6ddffcdd1a732567d5ec8b3d5749e3c7a5e4b",
    )
    manifest = {name: checksum(folder / name) for name in ("segmentation.onnx", "embedding.onnx")}
    (folder / "checksums.json").write_text(json.dumps(manifest, indent=2))
    print(f"Diarization ready: {folder}", flush=True)


# Optional offline Piper/VITS voices used only to synthesise the demo recording.
DEMO_VOICES = {
    "vits-piper-ru_RU-denis-medium": "efa4c18e0b5e32b81d1b6df36b9d312831e5d545200e27848ef926a4cd930300",
    "vits-piper-ru_RU-irina-medium": "1fc0f54e5e084fe287c07909f2f6e0ba6d857864cf800e3ab80286a4e8233008",
    "vits-piper-kk_KZ-issai-high": "6c59955f7f5e3fc50b130ced59a430eb4ef62d69155af4ce836db0dde4a42682",
}


def demo_voices(root: Path):
    folder = root / "tts"
    base = "https://github.com/k2-fsa/sherpa-onnx/releases/download/tts-models/"
    for name, sha in DEMO_VOICES.items():
        if (folder / name / "tokens.txt").is_file():
            print(f"Voice ready: {folder / name}", flush=True)
            continue
        archive = folder / f"{name}.tar.bz2"
        download(base + f"{name}.tar.bz2", archive, sha)
        with tarfile.open(archive, "r:bz2") as bundle:
            bundle.extractall(folder, filter="data")
        archive.unlink()
        print(f"Voice ready: {folder / name}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--whisper", choices=["tiny", "small", "medium", "large-v3"])
    parser.add_argument("--diarization", action="store_true")
    parser.add_argument(
        "--demo-voices",
        action="store_true",
        help="Optional offline TTS voices for scripts/make_demo_audio.py",
    )
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    if args.whisper:
        whisper(args.whisper, args.model_dir)
    if args.diarization:
        diarization(args.model_dir)
    if args.demo_voices:
        demo_voices(args.model_dir)
    if not args.whisper and not args.diarization and not args.demo_voices:
        parser.error("Select --whisper MODEL, --diarization and/or --demo-voices")


if __name__ == "__main__":
    main()
