"""Explicit, immutable model download. Standard library only; no runtime downloads."""

import argparse
import hashlib
import json
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


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--whisper", default="large-v3")
    parser.add_argument("--model-dir", type=Path, default=Path("models"))
    args = parser.parse_args()
    whisper(args.whisper, args.model_dir)


if __name__ == "__main__":
    main()
