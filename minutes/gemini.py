"""Optional CLOUD transcription through the official Gemini REST API (explicit opt-in only).

Audio sent here leaves the machine for Google. The API key is read from the environment,
sent only in the x-goog-api-key header and never logged or included in errors.
"""

import base64
import io
import json
import os
import re
import time
import wave
from pathlib import Path

import httpx

from minutes.models import Segment, Transcript

API = "https://generativelanguage.googleapis.com"
# gemini-3.5-transcribe was tested and rejected: it ignores JSON mode, returns no speakers or
# timestamps, and dropped the Kazakh lines of the demo. A general audio model is used instead.
DEFAULT_MODEL = "gemini-3.5-flash"
RETRY_STATUS = {429, 500, 503}
RETRY_DELAY = 5.0
INLINE_LIMIT = 14 * 1024 * 1024  # Requests above ~20 MB must use the Files API.

PROMPT = """Transcribe this meeting recording VERBATIM.
Speech is Russian, Kazakh, or mixed Russian/Kazakh with code-switching inside sentences.
Write every word in the language and script actually spoken: Russian in Russian Cyrillic,
Kazakh in Kazakh Cyrillic with Kazakh letters (ә ғ қ ң ө ұ ү һ і). Never translate,
summarize, paraphrase, correct grammar or merge speakers. Keep names, numbers and dates exactly
as said. Split into segments at every speaker change or pause. For each segment give the
speaker as "Speaker 1", "Speaker 2", ... in order of first appearance (the same voice keeps the
same label), and start/end in seconds from the beginning of the recording. If a part is
inaudible write [неразборчиво]. Return only JSON matching the schema."""

SCHEMA = {
    "type": "OBJECT",
    "properties": {
        "language": {"type": "STRING"},
        "segments": {
            "type": "ARRAY",
            "items": {
                "type": "OBJECT",
                "properties": {
                    "speaker": {"type": "STRING"},
                    "start": {"type": "NUMBER"},
                    "end": {"type": "NUMBER"},
                    "text": {"type": "STRING"},
                },
                "required": ["speaker", "start", "end", "text"],
            },
        },
    },
    "required": ["segments"],
}


def api_key() -> str:
    key = (os.getenv("GEMINI_API_KEY") or "").strip()
    if not key:
        raise ValueError(
            "GEMINI_API_KEY is not set. Add it to .env (see README) or use Local Whisper."
        )
    return key


def model_name() -> str:
    return (os.getenv("GEMINI_MODEL") or DEFAULT_MODEL).strip()


def to_wav(audio: Path) -> tuple[bytes, float]:
    """Decode any uploaded format locally to 16 kHz mono WAV, a MIME type Gemini accepts."""
    from faster_whisper.audio import decode_audio

    try:
        samples = decode_audio(str(audio), sampling_rate=16000)
    except Exception as exc:  # PyAV raises many decoder-specific types.
        raise ValueError("The recording could not be decoded. Upload a valid audio file.") from exc
    pcm = (samples.clip(-1, 1) * 32767).astype("<i2").tobytes()
    buffer = io.BytesIO()
    with wave.open(buffer, "wb") as output:
        output.setnchannels(1)
        output.setsampwidth(2)
        output.setframerate(16000)
        output.writeframes(pcm)
    return buffer.getvalue(), len(samples) / 16000


def error_message(response: httpx.Response) -> str:
    try:
        detail = response.json().get("error", {}).get("message", "")
    except ValueError:
        detail = ""
    return f"Gemini API error {response.status_code}: {detail[:300] or response.reason_phrase}"


def check(response: httpx.Response) -> httpx.Response:
    if response.is_error:
        raise RuntimeError(error_message(response))
    return response


def upload(client: httpx.Client, data: bytes) -> dict:
    start = check(
        client.post(
            "/upload/v1beta/files",
            headers={
                "X-Goog-Upload-Protocol": "resumable",
                "X-Goog-Upload-Command": "start",
                "X-Goog-Upload-Header-Content-Length": str(len(data)),
                "X-Goog-Upload-Header-Content-Type": "audio/wav",
            },
            json={"file": {"display_name": "meeting-audio"}},
        )
    )
    url = start.headers.get("x-goog-upload-url")
    if not url:
        raise RuntimeError("Gemini Files API did not return an upload URL")
    done = check(
        client.post(
            url,
            headers={"X-Goog-Upload-Offset": "0", "X-Goog-Upload-Command": "upload, finalize"},
            content=data,
        )
    )
    info = done.json()["file"]
    for _ in range(60):
        if info.get("state") != "PROCESSING":
            break
        time.sleep(2)
        info = check(client.get(f"/v1beta/{info['name']}")).json()
    if info.get("state") == "FAILED":
        raise RuntimeError("Gemini could not process the uploaded audio")
    return info


def speaker_label(raw: str, labels: dict[str, str]) -> str:
    key = raw.strip().casefold() or "unknown"
    if key not in labels:
        labels[key] = f"SPEAKER_{len(labels):02}"
    return labels[key]


def parse_response(data: dict, duration: float, model: str) -> Transcript:
    candidates = data.get("candidates") or []
    if not candidates:
        reason = data.get("promptFeedback", {}).get("blockReason", "no candidates")
        raise RuntimeError(f"Gemini returned no transcript ({reason})")
    candidate = candidates[0]
    if candidate.get("finishReason") == "MAX_TOKENS":
        raise RuntimeError("Gemini transcript was truncated. Split the recording and retry.")
    text = "".join(
        part.get("text", "")
        for part in candidate.get("content", {}).get("parts", [])
        if not part.get("thought")
    ).strip()
    text = re.sub(r"^```(?:json)?\s*|\s*```$", "", text)
    try:
        body = json.loads(text)
        rows = body["segments"]
    except (ValueError, KeyError, TypeError) as exc:
        raise RuntimeError("Gemini returned an unreadable transcript; retry.") from exc
    warnings = [
        "Cloud transcription by Google Gemini: speaker labels and timestamps are the "
        "model's estimates, not local diarization. Verify against the audio."
    ]
    labels: dict[str, str] = {}
    cleaned = []
    for row in rows:
        if not isinstance(row, dict) or not str(row.get("text", "")).strip():
            continue
        try:
            start, end = float(row.get("start")), float(row.get("end"))
        except (TypeError, ValueError):
            start = end = -1.0
        cleaned.append((str(row.get("speaker") or ""), start, end, str(row["text"]).strip()))
    # Timestamps are kept only if they are coherent; otherwise they are not invented.
    tolerance = duration + 2
    valid = all(0 <= s <= e <= tolerance for _, s, e, _ in cleaned) and all(
        a[1] <= b[1] + 1 for a, b in zip(cleaned, cleaned[1:])
    )
    if not valid:
        warnings.append("Gemini timestamps were inconsistent and are omitted (shown as 00:00:00).")
    segments = [
        Segment(
            id=index,
            start=min(start, duration) if valid else 0,
            end=min(end, duration) if valid else 0,
            text=text,
            speaker=speaker_label(speaker, labels),
        )
        for index, (speaker, start, end, text) in enumerate(cleaned)
    ]
    language = str(body.get("language") or "ru+kk")[:20] if isinstance(body, dict) else "ru+kk"
    return Transcript(
        segments=segments,
        language=language,
        duration=duration,
        device="cloud: Google Gemini API",
        model=model,
        warnings=warnings,
    )


def transcribe_gemini(audio: Path, model: str | None = None) -> Transcript:
    key = api_key()
    model = model or model_name()
    data, duration = to_wav(audio)
    if duration < 0.5:
        raise ValueError("No speech detected. Upload a recording with audible speech.")
    with httpx.Client(
        base_url=API,
        headers={"x-goog-api-key": key},
        timeout=httpx.Timeout(900, connect=15),
    ) as client:
        uploaded = None
        try:
            if len(data) > INLINE_LIMIT:
                uploaded = upload(client, data)
                audio_part = {"file_data": {"mime_type": "audio/wav", "file_uri": uploaded["uri"]}}
            else:
                audio_part = {
                    "inline_data": {
                        "mime_type": "audio/wav",
                        "data": base64.b64encode(data).decode(),
                    }
                }
            request = {
                "contents": [{"role": "user", "parts": [{"text": PROMPT}, audio_part]}],
                "generationConfig": {
                    "responseMimeType": "application/json",
                    "responseSchema": SCHEMA,
                    "maxOutputTokens": 32768,
                },
            }
            for attempt in range(3):
                response = client.post(f"/v1beta/models/{model}:generateContent", json=request)
                if response.status_code not in RETRY_STATUS or attempt == 2:
                    break
                time.sleep(RETRY_DELAY * (attempt + 1))  # Transient overload / rate limit.
            return parse_response(check(response).json(), duration, model)
        except httpx.HTTPError as exc:
            raise RuntimeError(f"Gemini API unreachable: {type(exc).__name__}") from exc
        finally:
            if uploaded:
                try:  # Do not leave meeting audio stored in the cloud project.
                    client.delete(f"/v1beta/{uploaded['name']}")
                except httpx.HTTPError:
                    pass
