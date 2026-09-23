# Implementation brief and evidence map

Source: participant-provided HackAlem AI 2026 case in this session. The initial
organizer repository contained only a two-line README; no separate case/scoring
documents were available. Work started 23 September 2026 at approximately 14:40
Astana time, within the 13:00–18:00 competition window.

Build a local/self-hosted meeting assistant for Russian, Kazakh and mixed speech.
No meeting audio or text may go to external inference services. Use Python 3.12,
Streamlit, faster-whisper, local Ollama Qwen3, sherpa-onnx and document exports.
Work directly on main in the organizer repository. Commit runnable milestones;
do not push without instruction. Preserve existing AGENTS.md and CLAUDE.md.

| Requirement / score | Implementation target | Observable verification |
| --- | --- | --- |
| STT; Russian, Kazakh, mixed speech | `minutes/transcription.py`, configurable multilingual Whisper | Upload real recordings; inspect timestamped transcript against speech |
| Responsible person, task, deadline | `minutes/extraction.py`, validated Pydantic JSON | Local extraction of explicit commitments; unknown information stays null |
| Speaker diarization and people linkage | `minutes/diarization.py`, overlap assignment, UI mapping | Two-speaker recording produces turns; mapped names appear in transcript/actions |
| Export protocol | `minutes/exports.py` | Open DOCX/PDF containing summary, action table, transcript and Cyrillic/Kazakh glyphs |
| Task compliance / workability: 25 | Complete upload-to-protocol flow | Local integration smoke and manual demo |
| Technical implementation: 25 | Separate typed modules, real models, retries/fallback | Core tests and real inference |
| README / reproducibility: 25 | Locked uv environment, model download script, exact runbook | Fresh environment install and documented verification |
| Value / applicability: 15 | Human review, speaker mapping, status dashboard, evidence links | Review/edit/export UI path |
| Development potential / originality: 10 | Reusable pipeline contracts and explicit privacy boundary | Architecture and future adapter design; no Teams/Zoom integrations now |

## Short implementation plan

1. **MVP:** scaffold Python 3.12 project; typed transcript/action schemas;
   isolated faster-whisper worker with CUDA-to-CPU fallback; loopback-only Ollama
   schema extraction with repair; Streamlit upload/transcript/actions; setup and
   model download instructions. Verify actual STT + local extraction and commit.
2. **Diarization:** public segmentation/embedding download script; real CPU ONNX
   diarization; timestamp overlap assignment; speaker-name mapping and regeneration.
3. **Protocol:** DOCX and Unicode PDF export; editable action statuses and due dates.
4. **Submission readiness:** parser/schema/core/UI/export tests; real integration
   smoke; clean uv startup; sources/licenses, limitations and exact verification.

## Decisions

- Sequential inference frees Whisper GPU memory before Ollama runs on the 8 GB GPU.
- Model downloads are explicit setup operations. Inference uses local files only.
- Ollama is restricted to loopback and local models; proxies/redirects are disabled.
- No participant names are guessed from voice. A person can be explicitly named in
  speech or assigned by the reviewer to an anonymous diarization label.
- Unknown deadlines/responsible people remain null; every extracted action cites
  transcript segment IDs and a verbatim supporting quote for human review.
- No cloud fallback and no fabricated demo responses. Small-model smoke results
  do not establish Kazakh or mixed-language recognition quality.
