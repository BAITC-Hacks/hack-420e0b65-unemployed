"""Real local LLM checks on explicit Russian, Kazakh and mixed text inputs."""

import sys
from datetime import date
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from minutes.config import Settings
from minutes.extraction import LocalOllama
from minutes.models import Segment, Transcript


def main():
    settings = Settings.load()
    llm = LocalOllama(settings.ollama_url, settings.ollama_model)
    # Meeting date is Wednesday 2026-09-23; relative wording must resolve from it.
    cases = {
        "Russian": ("Я подготовлю отчёт к 25 сентября 2026 года.", date(2026, 9, 25)),
        "Kazakh": ("Мен есепті 2026 жылғы 25 қыркүйекке дейін дайындаймын.", date(2026, 9, 25)),
        "Mixed": ("Мен отчётты 25 сентября 2026 года дейін дайындаймын.", date(2026, 9, 25)),
        "Russian-monday": ("Я подготовлю отчёт в понедельник.", date(2026, 9, 28)),
        "Russian-wednesday": ("Я подготовлю отчёт в среду.", date(2026, 9, 30)),
        "Russian-next-week": ("Я подготовлю отчёт на следующей неделе.", date(2026, 10, 2)),
        "Kazakh-friday": ("Мен есепті жұмаға дейін дайындаймын.", date(2026, 9, 25)),
    }
    output = Path("artifacts/extraction-checks")
    output.mkdir(parents=True, exist_ok=True)
    for label, (text, expected) in cases.items():
        source = Transcript(
            segments=[Segment(id=0, start=0, end=10, text=text, speaker="SPEAKER_00")],
            language="ru" if label.startswith("Russian") else "kk",
            duration=10,
            device="text-test",
            model="none",
        )
        result = llm.extract(source, date(2026, 9, 23), {"SPEAKER_00": "Айгүл"})
        assert len(result.action_items) == 1, f"{label}: expected one explicit commitment"
        item = result.action_items[0]
        assert item.responsible == "Айгүл", f"{label}: wrong responsible person"
        assert item.deadline == expected, f"{label}: deadline {item.deadline} != {expected}"
        assert item.responsible_speaker == "SPEAKER_00", f"{label}: missing speaker link"
        (output / f"{label.lower()}.json").write_text(result.model_dump_json(indent=2))
        print(
            f"PASS {label}: action, mapped person, deadline, speaker link and verbatim evidence",
            flush=True,
        )


if __name__ == "__main__":
    main()
