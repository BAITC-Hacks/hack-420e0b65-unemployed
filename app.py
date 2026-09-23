import tempfile
from datetime import date
from pathlib import Path

import httpx
import streamlit as st

from minutes.config import Settings
from minutes.diarization import assign_speakers, diarize
from minutes.exports import export_docx, export_pdf
from minutes.extraction import LocalOllama
from minutes.models import Meeting, timestamp, transcript_text
from minutes.review import apply_action_edits
from minutes.transcription import transcribe

st.set_page_config(
    page_title="Alem Minutes — local meeting assistant", page_icon="🎙️", layout="wide"
)
st.title("Alem Minutes")
st.caption("Локальный протокол встречи · Қазақша / Русский")
st.success(
    "**On-premise.** Audio, transcript and minutes never leave this machine: speech "
    "recognition and diarization run in local processes, and the language model is reached "
    "only over loopback. No cloud API, no account, no telemetry.",
    icon="🔒",
)
st.caption(
    "Demo flow: upload audio → 1. Transcribe locally (speakers included) → "
    "name the speakers → 2. Extract summary and action items → export DOCX/PDF."
)
try:
    settings = Settings.load()
except ValueError as exc:
    st.error(str(exc))
    st.stop()

with st.sidebar:
    st.header("Local models")
    model = st.selectbox(
        "Whisper model",
        list(dict.fromkeys([settings.whisper_model, "large-v3", "medium", "small", "tiny"])),
    )
    device = st.selectbox(
        "STT device",
        ["auto", "cuda", "cpu"],
        index=["auto", "cuda", "cpu"].index(settings.whisper_device),
    )
    language = st.selectbox("Speech language", ["Auto / mixed RU + KZ", "Russian", "Kazakh"])
    use_diarization = st.checkbox("Identify speaker turns (local ONNX)", value=True)
    num_speakers = st.number_input("Known number of speakers (0 = automatic)", 0, 20, 0)
    auto_extract = st.checkbox(
        "Extract minutes automatically after transcription",
        value=True,
        help="One-click demo. Turn off to map speaker names before the local model runs.",
    )
    st.caption(f"Ollama: {settings.ollama_model}\n\n{settings.ollama_url}")
    if st.button("Check local Ollama"):
        try:
            LocalOllama(settings.ollama_url, settings.ollama_model).check_local_model()
            st.success("Local model ready")
        except (httpx.HTTPError, ValueError) as exc:
            st.error(
                f"Local Ollama unavailable: {exc}. Run `ollama serve` and "
                f"`ollama pull {settings.ollama_model}`."
            )
    st.info(
        "Download models once using the README. Runtime never downloads models or calls cloud inference."
    )


def run_extraction(meeting) -> None:
    """Single place for local LLM minutes so the button and the one-click flow behave alike."""
    try:
        with st.spinner("Extracting summary and action items using local Ollama…"):
            meeting.extraction = LocalOllama(settings.ollama_url, settings.ollama_model).extract(
                meeting.transcript, meeting.meeting_date, meeting.speaker_names
            )
            st.session_state["editor_revision"] = st.session_state.get("editor_revision", 0) + 1
    except httpx.HTTPError as exc:
        st.error(
            f"Local Ollama did not answer; the transcript is preserved. {exc}\n\n"
            f"Start it with `ollama serve` and `ollama pull {settings.ollama_model}`."
        )
    except (ValueError, OSError) as exc:
        st.error(f"Extraction failed; transcript is preserved. {exc}")


title = st.text_input("Meeting title", "Рабочая встреча")
meeting_date = st.date_input("Meeting date (for relative deadlines)", date.today())
upload = st.file_uploader(
    "Upload meeting audio",
    type=["wav", "mp3", "mpeg", "m4a", "ogg", "flac", "mp4"],
    on_change=lambda: st.session_state.pop("meeting", None),
)
if upload:
    st.audio(upload)

if st.button("1. Transcribe locally", type="primary", disabled=upload is None):
    st.session_state.pop("meeting", None)
    if not title.strip():
        st.error("Enter a meeting title.")
    else:
        try:
            with st.spinner("Transcribing locally. First CUDA attempt may fall back to CPU…"):
                with tempfile.TemporaryDirectory(prefix="alem-audio-") as folder:
                    audio = Path(folder) / f"recording{Path(upload.name).suffix.lower()}"
                    audio.write_bytes(upload.getvalue())
                    transcript = transcribe(
                        audio,
                        settings.model_dir,
                        model,
                        device,
                        {"Russian": "ru", "Kazakh": "kk"}.get(language),
                    )
                    turns = []
                    if use_diarization and transcript.segments:
                        try:
                            with st.spinner("Finding speaker turns locally…"):
                                turns = diarize(audio, settings.model_dir, int(num_speakers))
                                transcript = assign_speakers(transcript, turns)
                                if not turns:
                                    transcript.warnings.append(
                                        "No speaker turns detected; identities remain unknown."
                                    )
                        except (RuntimeError, ValueError, OSError) as exc:
                            transcript.warnings.append(f"Diarization unavailable: {exc}")
                st.session_state.meeting = Meeting(
                    title=title,
                    meeting_date=meeting_date,
                    transcript=transcript,
                    speaker_turns=turns,
                )
            if auto_extract and st.session_state.meeting.transcript.segments:
                run_extraction(st.session_state.meeting)
        except (RuntimeError, ValueError, OSError) as exc:
            st.error(str(exc))

meeting = st.session_state.get("meeting")
if meeting:
    st.caption(f"Current result: {meeting.title} · {meeting.meeting_date}")
    for warning in meeting.transcript.warnings:
        st.warning(warning)
    st.caption(
        f"Model: {meeting.transcript.model} · Device: {meeting.transcript.device} · "
        f"Detected language: {meeting.transcript.language} · {meeting.transcript.duration:.1f} seconds"
    )
    speakers = sorted({s.speaker for s in meeting.transcript.segments if s.speaker})
    if speakers:
        st.subheader("Who is speaking?")
        st.caption(
            "Voice clusters do not reveal real identities. Listen and map labels to participant names."
        )
        with st.form("speaker_names"):
            names = {}
            for speaker in speakers:
                sample = next(s for s in meeting.transcript.segments if s.speaker == speaker)
                value = st.text_input(
                    f"{speaker} · {timestamp(sample.start)} · {sample.text[:80]}",
                    value=meeting.speaker_names.get(speaker, ""),
                    max_chars=100,
                )
                if value.strip():
                    names[speaker] = value.strip()
            if st.form_submit_button("Apply participant names"):
                if names != meeting.speaker_names:
                    meeting.speaker_names = names
                    meeting.extraction = None
                st.rerun()
        with st.expander("Speaker timeline"):
            st.dataframe([t.model_dump() for t in meeting.speaker_turns], hide_index=True)
    st.subheader("Timestamped transcript")
    text = transcript_text(meeting.transcript, meeting.speaker_names)
    st.text_area("Transcript", text, height=280, disabled=True)
    st.download_button("Download transcript", text, "transcript.txt", "text/plain")
    if not meeting.transcript.segments:
        st.warning("No speech detected. Try a clearer recording.")
    if st.button("2. Extract summary and action items", disabled=not meeting.transcript.segments):
        run_extraction(meeting)
    if meeting.extraction:
        st.subheader("Meeting summary")
        st.write(meeting.extraction.summary)
        st.subheader("Action items")
        st.caption("AI draft — verify people, deadlines and supporting quotes before use.")
        rows = [
            {
                "Responsible": a.responsible or "Not specified",
                "Task": a.task,
                "Deadline": a.display_deadline(),
                "Status": a.display_status(),
                "Evidence": a.evidence_quote,
                "Segments": ", ".join(map(str, a.evidence_segment_ids)),
            }
            for a in meeting.extraction.action_items
        ]
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True)
            statuses = [a.display_status() for a in meeting.extraction.action_items]
            for column, label in zip(st.columns(3), ["in progress", "overdue", "completed"]):
                column.metric(label.title(), statuses.count(label))
            with st.expander("Review actions and update progress"):
                st.caption(
                    "Overdue is calculated from the date. Edits are included in all exports."
                )
                with st.form("review_actions"):
                    editable = [
                        {
                            "Responsible": a.responsible or "",
                            "Task": a.task,
                            "Deadline": a.deadline,
                            "Status": a.status,
                        }
                        for a in meeting.extraction.action_items
                    ]
                    edits = st.data_editor(
                        editable,
                        hide_index=True,
                        width="stretch",
                        num_rows="fixed",
                        key=f"actions_{st.session_state.get('editor_revision', 0)}",
                        column_config={
                            "Deadline": st.column_config.DateColumn(format="YYYY-MM-DD"),
                            "Status": st.column_config.SelectboxColumn(
                                options=["in progress", "completed"], required=True
                            ),
                            "Task": st.column_config.TextColumn(required=True),
                        },
                    )
                    if st.form_submit_button("Save action updates"):
                        try:
                            meeting.extraction.action_items = apply_action_edits(
                                meeting.extraction.action_items, edits
                            )
                            st.session_state["editor_revision"] = (
                                st.session_state.get("editor_revision", 0) + 1
                            )
                            st.rerun()
                        except (ValueError, TypeError) as exc:
                            st.error(f"Invalid edit: {exc}")
        else:
            st.info("No explicit action items found.")
        st.subheader("Export protocol")
        try:
            docx_column, pdf_column = st.columns(2)
            docx_column.download_button(
                "Download DOCX",
                export_docx(meeting),
                "meeting-protocol.docx",
                "application/vnd.openxmlformats-officedocument.wordprocessingml.document",
            )
            pdf_column.download_button(
                "Download PDF", export_pdf(meeting), "meeting-protocol.pdf", "application/pdf"
            )
        except (ValueError, OSError) as exc:
            st.error(f"Document export failed; JSON and transcript remain available. {exc}")
        st.download_button(
            "Download protocol JSON",
            meeting.model_dump_json(indent=2),
            "meeting.json",
            "application/json",
        )

if st.sidebar.button("Clear meeting from session"):
    st.session_state.clear()
    st.rerun()
