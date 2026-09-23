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

## Milestone 3 (approximately 15:25 Astana)

- Full suite: **31 passed**, including actual DOCX XML content, PDF text extraction
  for Russian and all nine Kazakh-specific letters, a 100-segment multipage PDF,
  action review validation and Streamlit dashboard/export rendering (including
  a task with an unknown deadline). Ruff and Python compilation passed.
- Real Kazakh audio → large-v3 CUDA → real diarization → local Qwen → DOCX/PDF/JSON:
  completed using `scripts/smoke.py artifacts/kazakh-public-sample.mp3 --model
  large-v3 --language kk --diarize --output artifacts/kazakh-smoke.json`.
- Source: [public Kazakh Piper/ISSAI synthetic speech sample, speaker 0](https://k2-fsa.github.io/sherpa/onnx/tts/all/Kazakh/vits-piper-kk_KZ-issai-high.html).
  This is externally published test audio downloaded for testing; no application
  audio/text was sent out. It is not a human meeting benchmark.
- Recognition limitation observed: reference “Әлемнің жұлдыздары сенің көзің,
  жаным.” was transcribed “Әлімнің жолдыздары сенің көзің жаным.” Two words differ;
  **do not claim perfect Kazakh recognition**. The sample contains no commitments;
  zero extracted actions is expected. Human review remains necessary.

## Milestone 4 — relative deadlines and speaker attribution (approximately 16:20 Astana)

Commits before this entry (`62998eb` … `91984e2`) added diarization, exports,
review and hardening; the full suite was 47 tests at `91984e2`. Latest verified run:

- `uv run pytest -q`: **79 passed** (32 new in `tests/test_deadlines.py`).
  `uv run ruff check .` and `uv run ruff format --check .`: passed.
- Relative deadlines are now resolved deterministically in `minutes/deadlines.py`
  from the meeting date (the local model was previously the only weekday counter).
  Unit-tested for meeting date Wednesday 2026-09-23: “в понедельник” → 09-28,
  “в среду” → 09-30 (same weekday = next week), “в эту среду” → 09-23,
  “в следующую пятницу” → 10-02, “на следующей неделе” → 10-02 (Friday of that
  week), “до конца недели” / “осы аптаның соңына дейін” → 09-25, “завтра” / “ертең”,
  “послезавтра”, “через неделю”, “жұмаға дейін”, “дүйсенбіге дейін”, “келесі аптада”.
  Absolute dates (“25 сентября”) stay with the model; “в среде разработки” is not
  treated as Wednesday.
- Speaker attribution: a task keeps `responsible_speaker` only if that speaker voiced a
  cited evidence segment; a link to the adjacent speaker (e.g. the one who asked the
  question) is dropped, and a label-only assignee is cleared, while an explicitly named
  assignee is kept. Words in a ≤0.3 s pause between diarization turns now join the
  nearer turn instead of becoming `UNKNOWN` fragments; equidistant words stay unknown.
  Covered by unit tests with adjacent, overlapping and gap turns.
- `uv run python scripts/verify_extraction.py` (real local Qwen
  `qwen3:4b-instruct-2507-q4_K_M`, meeting date 2026-09-23): **7 PASS** — Russian,
  Kazakh, mixed explicit date → 2026-09-25; “в понедельник” → 2026-09-28; “в среду”
  → 2026-09-30; “на следующей неделе” → 2026-10-02; “жұмаға дейін” → 2026-09-25.
- `uv run python scripts/smoke.py assets/demo/planning-ru-kk.wav --model large-v3
  --diarize --output artifacts/final-demo.json`: 8 segments, CUDA, `ru+kk`, 7 turns /
  3 speakers, 5 validated actions. Deadlines: “к 25 сентября 2026 года” → 09-25
  (SPEAKER_01), “до конца недели” → 09-25 (assigned by name, no speaker link),
  “осаптаның соңына дейін” → 09-25 (SPEAKER_02), “в понедельник” → 09-28 (SPEAKER_01),
  one Kazakh budget task without a deadline.
- Observed imperfection on the synthetic demo: ASR garbles some Kazakh words
  (“Жарайды, мен” → “Жар айтмейін”) and the model then used “Жар” as a responsible
  name for SPEAKER_02 before names were mapped. Mapping speaker names in the UI
  replaces it; human review remains required.
