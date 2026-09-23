from minutes.diarization import SpeakerTurn, assign_speakers

from minutes.models import Segment, Transcript, Word


def make_transcript(segments):
    return Transcript(segments=segments, duration=10, language="ru", device="cpu", model="test")


def test_overlap_uses_maximum_duration_not_nearest_start():
    source = make_transcript([Segment(id=0, start=2, end=6, text="Длинный ответ")])
    turns = [
        SpeakerTurn(start=0, end=2.5, speaker="SPEAKER_00"),
        SpeakerTurn(start=2.5, end=8, speaker="SPEAKER_01"),
    ]
    result = assign_speakers(source, turns)
    assert result.segments[0].speaker == "SPEAKER_01"
    assert source.segments[0].speaker is None


def test_no_overlap_keeps_unknown_instead_of_guessing():
    source = make_transcript([Segment(id=0, start=5, end=6, text="Ответ")])
    result = assign_speakers(source, [SpeakerTurn(start=0, end=2, speaker="SPEAKER_00")])
    assert result.segments[0].speaker is None


def test_splits_whisper_segment_when_speaker_changes_between_words():
    source = make_transcript(
        [
            Segment(
                id=0,
                start=0,
                end=4,
                text="Вопрос? Да.",
                words=[
                    Word(start=0, end=1, text="Вопрос?"),
                    Word(start=3, end=4, text=" Да."),
                ],
            )
        ]
    )
    turns = [
        SpeakerTurn(start=0, end=1.5, speaker="SPEAKER_00"),
        SpeakerTurn(start=2.5, end=4, speaker="SPEAKER_01"),
    ]
    result = assign_speakers(source, turns)
    assert [s.speaker for s in result.segments] == ["SPEAKER_00", "SPEAKER_01"]
    assert [s.text for s in result.segments] == ["Вопрос?", "Да."]
    assert [s.id for s in result.segments] == [0, 1]


def test_equal_overlapping_speakers_are_marked_uncertain():
    source = make_transcript([Segment(id=0, start=1, end=2, text="Одновременно")])
    turns = [SpeakerTurn(start=0, end=3, speaker=s) for s in ["SPEAKER_00", "SPEAKER_01"]]
    assert assign_speakers(source, turns).segments[0].speaker is None
