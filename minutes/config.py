import os
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import urlparse

from dotenv import load_dotenv


def local_ollama_url(value: str) -> str:
    parsed = urlparse(value)
    if (
        parsed.scheme != "http"
        or parsed.hostname not in {"127.0.0.1", "localhost", "::1"}
        or parsed.username
        or parsed.password
        or parsed.path not in {"", "/"}
        or parsed.query
        or parsed.fragment
    ):
        raise ValueError("OLLAMA_URL must be a local loopback HTTP address")
    # Resolve localhost ourselves to avoid proxy/DNS configuration sending meeting text elsewhere.
    host = "[::1]" if parsed.hostname == "::1" else "127.0.0.1"
    return f"http://{host}:{parsed.port or 11434}"


@dataclass(frozen=True)
class Settings:
    model_dir: Path
    whisper_model: str
    whisper_device: str
    ollama_url: str
    ollama_model: str

    @classmethod
    def load(cls):
        load_dotenv()
        return cls(
            model_dir=Path(os.getenv("MODEL_DIR", "models")),
            whisper_model=os.getenv("WHISPER_MODEL", "large-v3"),
            whisper_device=os.getenv("WHISPER_DEVICE", "auto"),
            ollama_url=local_ollama_url(os.getenv("OLLAMA_URL", "http://127.0.0.1:11434")),
            ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:4b-instruct-2507-q4_K_M"),
        )
