# Alem Minutes — local Russian/Kazakh meeting assistant

HackAlem AI 2026, team Unemployed. Upload audio, transcribe locally, and extract a
summary and action items (person, task, deadline, supporting quote) with local Qwen.
Russian, Kazakh and mixed speech use multilingual Whisper; quality needs checking
on representative recordings. No cloud inference or personal accounts required.

## Current milestone

Streamlit upload, timestamped faster-whisper STT, validated Ollama extraction with
retries, transcript/JSON downloads. Diarization and document export are next.
Actual verification results will be recorded in `docs/VERIFICATION.md`.

## Setup — Ubuntu / WSL2

Requirements: uv, local Ollama, space for model weights; internet only for setup.
Python 3.12 is managed by uv, not system Python 3.14. PyAV wheels decode audio
without a separate ffmpeg executable. GPU is optional; CPU inference is slower.

```bash
uv sync --python 3.12 --extra gpu
cp .env.example .env
ollama pull qwen3:4b-instruct-2507-q4_K_M
uv run python scripts/download_models.py --whisper large-v3
uv run streamlit run app.py
```

Open http://127.0.0.1:8501. Start `ollama serve` separately if needed.
CPU-only install: `uv sync --python 3.12`. For less memory, download
`--whisper medium`, then select medium in the sidebar. `--whisper tiny` is a fast
pipeline smoke test only, not a Kazakh quality baseline.

The `gpu` extra provides CUDA 12 cuBLAS and cuDNN 9. The isolated worker configures
their loader paths, attempts CUDA, and retries on CPU int8 if GPU inference fails.
It exits before Ollama runs, freeing its GPU memory.

## Verify the main scenario

1. Upload WAV, MP3/MPEG, M4A, OGG, FLAC or MP4 (up to 200 MB).
2. Enter title/date. Auto language mode enables multilingual/code-switching;
   Russian/Kazakh selection is available for monolingual recordings.
3. Click **1. Transcribe locally**. Expect timestamps, text, actual device, or a
   clear missing-model/invalid-audio error.
4. Click **2. Extract summary and action items**. Expect summary and action table
   with source quotes/IDs. Missing people/deadlines stay unspecified. No tasks is
   a valid result. Review every AI draft against audio.
5. Download transcript or JSON. Clear meeting from the session when finished.

For a spoken test, record “Айгуль, подготовь отчёт к 25 сентября 2026 года.
Хорошо, я подготовлю отчёт к 25 сентября 2026 года.” Check the transcription,
task and date. This is a test input, not hardcoded output. Test Kazakh and mixed
recordings separately; a successful Russian test does not establish their quality.

```bash
uv run pytest -q
uv run ruff check .
```

## Architecture / privacy

Streamlit → temporary local audio → isolated faster-whisper → typed Transcript →
loopback Ollama → validated Extraction → UI/downloads.

`minutes/models.py`: Pydantic contracts; `transcription.py`: local STT worker;
`extraction.py`: local LLM/schema/evidence boundary; `config.py`: settings;
`app.py`: UI. Future conferencing adapters can feed audio into these same modules.
Teams/Zoom/Meet integrations are not implemented.

Inference uses only pre-downloaded files. Ollama is restricted to loopback;
external hosts, HTTP proxies, redirects and cloud/remote models are rejected.
Streamlit telemetry is disabled. Audio is held in browser/session memory and a
temporary directory removed after processing; results remain in the session until
cleared. Downloads persist wherever the reviewer saves them. The app binds to
localhost and has no authentication. An internal multi-user deployment needs an
authenticated gateway. Air-gapped deployments should pre-download artifacts and
deny outbound networking, including from Ollama.

See `.env.example`: WHISPER_MODEL, WHISPER_DEVICE, MODEL_DIR, OLLAMA_URL and
OLLAMA_MODEL. No secrets needed. Initial model downloads are explicit setup steps.

## Limitations

- Schema and verbatim evidence checks do not establish semantic correctness.
- Long transcripts are chunked; cross-boundary commitments may lose context.
  Summary is currently capped at 6000 characters.
- Language detection does not prove mixed-language transcription quality.
- MVP speaker labels and PDF/DOCX export are not implemented yet.

## Third-party disclosure

Team work: pipeline integration, UI, validation and verification. Models/libraries
are third-party materials. Exact dependency versions are in `uv.lock`.

| Component | Source | License |
| --- | --- | --- |
| faster-whisper / CTranslate2 | [SYSTRAN](https://github.com/SYSTRAN/faster-whisper) / [OpenNMT](https://github.com/OpenNMT/CTranslate2) | MIT |
| Whisper / converted weights | [Whisper](https://github.com/openai/whisper) / [Systran large-v3](https://huggingface.co/Systran/faster-whisper-large-v3) | MIT |
| Qwen3-4B-Instruct-2507 | [Qwen](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) | Apache-2.0 |
| Ollama | [Ollama](https://github.com/ollama/ollama) | MIT |
| Streamlit | [Streamlit](https://github.com/streamlit/streamlit) | Apache-2.0 |
| Pydantic / HTTPX / dotenv | [Pydantic](https://github.com/pydantic/pydantic) / [HTTPX](https://github.com/encode/httpx) / [dotenv](https://github.com/theskumar/python-dotenv) | MIT / BSD-3-Clause / BSD-3-Clause |
| PyAV / bundled FFmpeg | [PyAV](https://github.com/PyAV-Org/PyAV) | BSD-3-Clause; bundled FFmpeg LGPL/GPL terms per distribution |
| Optional CUDA libraries | [NVIDIA](https://docs.nvidia.com/cuda/) | NVIDIA redistribution terms |

See `docs/BRIEF.md` for requirements/scoring map. Respect component/model licenses
when redistributing the application and its downloaded dependencies.
