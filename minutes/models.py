from datetime import date
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)


class Word(StrictModel):
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str


class Segment(StrictModel):
    id: int = Field(ge=0)
    start: float = Field(ge=0)
    end: float = Field(ge=0)
    text: str = Field(min_length=1)
    speaker: str | None = None
    words: list[Word] = Field(default_factory=list)

    @model_validator(mode="after")
    def ordered(self):
        if self.end < self.start:
            raise ValueError("Segment end must be after start")
        return self


class Transcript(StrictModel):
    segments: list[Segment]
    language: str
    duration: float = Field(ge=0)
    device: str
    model: str
    warnings: list[str] = Field(default_factory=list)


class ActionItem(StrictModel):
    task: str = Field(min_length=1, max_length=2000)
    responsible: str | None
    responsible_speaker: str | None
    deadline: date | None
    deadline_text: str | None
    evidence_segment_ids: list[int] = Field(min_length=1)
    evidence_quote: str = Field(min_length=1)
    status: Literal["in progress", "completed"] = "in progress"

    def display_status(self, today: date | None = None) -> str:
        if self.status == "completed":
            return "completed"
        if self.deadline and self.deadline < (today or date.today()):
            return "overdue"
        return "in progress"


class Extraction(StrictModel):
    summary: str = Field(min_length=1, max_length=6000)
    action_items: list[ActionItem]


class Meeting(StrictModel):
    title: str = Field(min_length=1, max_length=200)
    meeting_date: date
    transcript: Transcript
    extraction: Extraction | None = None
    speaker_names: dict[str, str] = Field(default_factory=dict)


def timestamp(seconds: float) -> str:
    total = int(seconds)
    return f"{total // 3600:02}:{total // 60 % 60:02}:{total % 60:02}"


def transcript_text(transcript: Transcript, names: dict[str, str] | None = None) -> str:
    names = names or {}
    return "\n".join(
        f"[{s.id}] {timestamp(s.start)}–{timestamp(s.end)} "
        f"{names.get(s.speaker, s.speaker) or 'UNKNOWN'}: {s.text}"
        for s in transcript.segments
    )
