from datetime import date
from io import BytesIO
from zipfile import ZipFile

from pypdf import PdfReader

from minutes.exports import export_docx, export_pdf
from minutes.models import ActionItem, Extraction, Meeting, Segment, Transcript


def example_meeting():
    return Meeting(
        title="Жиналыс & Встреча <2026>",
        meeting_date=date(2026, 9, 23),
        transcript=Transcript(
            segments=[
                Segment(
                    id=0,
                    start=0,
                    end=5,
                    speaker="SPEAKER_00",
                    text="Ә Ғ Қ Ң Ө Ұ Ү Һ І. Я подготовлю отчёт.",
                )
            ],
            language="kk",
            duration=5,
            model="test fixture",
            device="cpu",
        ),
        speaker_names={"SPEAKER_00": "Айгүл"},
        extraction=Extraction(
            summary="Обсудили отчёт & сроки <проверить>.",
            action_items=[
                ActionItem(
                    task="Подготовить есеп",
                    responsible="Айгүл",
                    responsible_speaker="SPEAKER_00",
                    deadline=date(2026, 9, 25),
                    deadline_text="25 сентября",
                    evidence_segment_ids=[0],
                    evidence_quote="Я подготовлю отчёт.",
                )
            ],
        ),
    )


def test_docx_contains_summary_tasks_speakers_and_transcript():
    output = export_docx(example_meeting())
    with ZipFile(BytesIO(output)) as archive:
        xml = archive.read("word/document.xml").decode()
    for text in [
        "Айгүл",
        "Подготовить есеп",
        "2026-09-25",
        "Ә Ғ Қ Ң Ө Ұ Ү Һ І",
        "Я подготовлю отчёт",
    ]:
        assert text in xml
    assert "&lt;проверить&gt;" in xml


def test_pdf_preserves_russian_and_all_kazakh_specific_letters():
    output = export_pdf(example_meeting())
    assert output.startswith(b"%PDF")
    pdf = PdfReader(BytesIO(output))
    text = "\n".join(page.extract_text() for page in pdf.pages)
    for expected in ["Айгүл", "Подготовить есеп", "2026-09-25", "Ә Ғ Қ Ң Ө Ұ Ү Һ І", "<проверить>"]:
        assert expected in text


def test_unresolved_deadline_shows_dash_not_spoken_wording():
    meeting = example_meeting()
    item = meeting.extraction.action_items[0]
    item.deadline, item.deadline_text = None, "когда-нибудь потом"
    with ZipFile(BytesIO(export_docx(meeting))) as archive:
        xml = archive.read("word/document.xml").decode()
    pdf = PdfReader(BytesIO(export_pdf(meeting)))
    pdf_text = "\n".join(page.extract_text() for page in pdf.pages)
    for text in (xml, pdf_text):
        assert "—" in text
        assert "когда-нибудь потом" not in text


def test_long_protocol_paginates_without_losing_last_segment():
    meeting = example_meeting()
    meeting.transcript.segments = [
        Segment(id=i, start=i * 3, end=i * 3 + 2, text=f"Строка {i}: " + "Обсудили задачу. " * 20)
        for i in range(100)
    ]
    pdf = PdfReader(BytesIO(export_pdf(meeting)))
    assert len(pdf.pages) > 1
    assert "Строка 99" in "\n".join(page.extract_text() for page in pdf.pages)
