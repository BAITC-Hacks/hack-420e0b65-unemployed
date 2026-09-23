import tempfile
from datetime import date
from pathlib import Path

import httpx
import streamlit as st

from minutes.config import Settings
from minutes.extraction import LocalOllama
from minutes.models import Meeting, transcript_text
from minutes.transcription import transcribe

st.set_page_config(
    page_title="Alem Minutes — local meeting assistant", page_icon="🎙️", layout="wide"
)
st.title("Alem Minutes")
st.caption("Локальный протокол встречи · Қазақша / Русский · Audio and text stay on this machine")
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
    st.caption(f"Ollama: {settings.ollama_model}\n\n{settings.ollama_url}")
    if st.button("Check local Ollama"):
        try:
            LocalOllama(settings.ollama_url, settings.ollama_model).check_local_model()
            st.success("Local model ready")
        except (httpx.HTTPError, ValueError) as exc:
            st.error(f"Local Ollama unavailable: {exc}")
    st.info(
        "Download models once using the README. Runtime never downloads models or calls cloud inference."
    )

title = st.text_input("Meeting title", "Рабочая встреча")
meeting_date = st.date_input("Meeting date (for relative deadlines)", date.today())
upload = st.file_uploader(
    "Upload meeting audio", type=["wav", "mp3", "mpeg", "m4a", "ogg", "flac", "mp4"]
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
                st.session_state.meeting = Meeting(
                    title=title,
                    meeting_date=meeting_date,
                    transcript=transcript,
                )
        except (RuntimeError, ValueError, OSError) as exc:
            st.error(str(exc))

meeting = st.session_state.get("meeting")
if meeting:
    for warning in meeting.transcript.warnings:
        st.warning(warning)
    st.caption(
        f"Model: {meeting.transcript.model} · Device: {meeting.transcript.device} · "
        f"Detected language: {meeting.transcript.language} · {meeting.transcript.duration:.1f} seconds"
    )
    st.subheader("Timestamped transcript")
    text = transcript_text(meeting.transcript, meeting.speaker_names)
    st.text_area("Transcript", text, height=280, disabled=True)
    st.download_button("Download transcript", text, "transcript.txt", "text/plain")
    if not meeting.transcript.segments:
        st.warning("No speech detected. Try a clearer recording.")
    if st.button("2. Extract summary and action items", disabled=not meeting.transcript.segments):
        try:
            with st.spinner("Extracting using local Ollama…"):
                result = LocalOllama(settings.ollama_url, settings.ollama_model).extract(
                    meeting.transcript,
                    meeting.meeting_date,
                    meeting.speaker_names,
                )
                meeting.extraction = result
        except (httpx.HTTPError, ValueError, OSError) as exc:
            st.error(f"Extraction failed; transcript is preserved. {exc}")
    if meeting.extraction:
        st.subheader("Meeting summary")
        st.write(meeting.extraction.summary)
        st.subheader("Action items")
        st.caption("AI draft — verify people, deadlines and supporting quotes before use.")
        rows = [
            {
                "Responsible": a.responsible or "Not specified",
                "Task": a.task,
                "Deadline": str(a.deadline or a.deadline_text or "Not specified"),
                "Status": a.display_status(),
                "Evidence": a.evidence_quote,
                "Segments": ", ".join(map(str, a.evidence_segment_ids)),
            }
            for a in meeting.extraction.action_items
        ]
        if rows:
            st.dataframe(rows, width="stretch", hide_index=True)
        else:
            st.info("No explicit action items found.")
        st.download_button(
            "Download protocol JSON",
            meeting.model_dump_json(indent=2),
            "meeting.json",
            "application/json",
        )

if st.sidebar.button("Clear meeting from session"):
    st.session_state.clear()
    st.rerun()
