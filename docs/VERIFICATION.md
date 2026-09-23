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

## Milestone 2 (approximately 15:20 Astana)

- Real `large-v3` Russian smoke using `scripts/smoke.py ... --model large-v3
  --language ru --diarize`: CUDA transcription, one real speaker turn, validated
  local Qwen summary. Same public 3.55-second Russian clip as milestone 1.
- Real CPU check: `scripts/smoke.py ... --model tiny --device cpu --language ru`:
  actual CPU transcription and local Qwen extraction passed.
- `python scripts/verify_extraction.py`: real local Qwen passed Russian, Kazakh,
  and mixed text commitments. Each result had one task, mapped person Айгүл,
  explicit date 2026-09-25, SPEAKER_00 link and a validated verbatim quote. These
  are text-level checks; not evidence of Kazakh speech recognition quality.
- Real diarization: `python -m minutes.diarization --audio artifacts/four-speakers.wav
  --model-dir models/diarization --speakers 4 --output artifacts/diarization-four.json`:
  10 turns across 4 labels. The known speaker count was supplied, not inferred.
  Source: [sherpa-onnx four-speaker Chinese test audio](https://github.com/k2-fsa/sherpa-onnx/releases/download/speaker-segmentation-models/0-four-speakers-zh.wav).
  This checks native model execution and clustering, not Russian/Kazakh diarization accuracy.
- Dependency discovery: sherpa-onnx 1.13.8 required a separate matching
  `sherpa-onnx-core` native library. Both are now explicit locked dependencies;
  native import is covered by a test.
- Speaker assignment tests cover maximum overlap, unknown spans, word-level
  splitting at speaker changes, and equal-overlap ambiguity. UI test checks name
  mapping updates transcript and invalidates previously extracted minutes.

Model files stay ignored. Whisper weights are pinned by upstream commit and SHA-256;
diarization release artifacts are pinned by SHA-256 in the download script.
