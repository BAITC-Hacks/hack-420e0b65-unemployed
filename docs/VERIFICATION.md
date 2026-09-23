# Verification log — 23 September 2026

## Milestone 1 (approximately 15:13 Astana)

- Installed CPython 3.12.14 through uv in `.python/`, and locked dependencies in
  `uv.lock`. Command: `uv sync --python 3.12 --extra gpu` (task-local uv cache and
  Python install directory were used by the agent). CUDA 12 cuBLAS/cuDNN 9 installed.
- `python -m pytest -q tests/test_core.py tests/test_extraction.py
  tests/test_transcription.py tests/test_ui.py`: **19 passed**. Covers schema,
  evidence, local-only endpoint/model boundary, bounded repair, chunk preservation,
  statuses, native-crash CPU retry (simulated) and Streamlit startup.
- `ruff check .`: passed after import formatting.
- `python scripts/download_models.py --whisper tiny`: downloaded pinned weights,
  checked SHA-256. No credentials used.
- Real integration: `python scripts/smoke.py
  artifacts/spoken-language-identification-test-wavs/ru-russian.wav --model tiny
  --language ru --output artifacts/mvp-smoke.json`: **passed**, actual device CUDA,
  one timestamped Russian segment, local Qwen summary, zero action items (this clip
  contains no meeting commitment). No cloud inference used.
- Test audio source: [sherpa-onnx public language-identification examples](https://github.com/k2-fsa/sherpa-onnx/releases/download/asr-models/spoken-language-identification-test-wavs.tar.bz2).
  Downloaded under ignored `artifacts/`; not redistributed as original team work.

Limitations at this checkpoint: tiny-model short Russian pipeline smoke is not a
meeting-quality benchmark. Kazakh/mixed speech, large-v3 and real diarization are
not verified at this checkpoint. CPU crash fallback unit test simulates the native
crash; actual CPU inference is still pending. No document exports yet.
