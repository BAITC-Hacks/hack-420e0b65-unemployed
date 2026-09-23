"""Deterministic relative-deadline resolution against the meeting date (RU/KK).

Small local models often miscount weekdays, so spoken relative deadlines are resolved in
code. Anything not matched here is left to the model, which must return null when unsure.
"""

import re
from datetime import date, timedelta

# Weekday index (Monday=0) -> inflected Russian and Kazakh forms.
WEEKDAYS = {
    0: r"понедельник[аеу]?|дүйсенбі\w*",
    1: r"вторник[аеу]?|сейсенбі\w*",
    2: r"сред[аыуе]|сәрсенбі\w*",
    3: r"четверг[аеу]?|бейсенбі\w*",
    4: r"пятниц[аыуе]|жұма\w*",
    5: r"суббот[аыуе]|сенбі\w*",
    6: r"воскресень[еяю]|жексенбі\w*",
}
NEXT = r"следующ\w*|будущ\w*|келесі"
THIS = r"эт\w+|бұл|осы"


def _word(pattern: str) -> str:
    return rf"(?<!\w)(?:{pattern})(?!\w)"


def resolve_deadline(text: str | None, meeting_date: date) -> date | None:
    """Return the calendar date for a spoken relative deadline, or None if not certain."""
    if not text:
        return None
    lowered = text.casefold().replace("ё", "е")
    for weekday, forms in WEEKDAYS.items():
        match = re.search(rf"(?:(?P<mod>{NEXT}|{THIS})\s+)?{_word(forms)}", lowered)
        if not match:
            continue
        mod = match.group("mod")
        ahead = (weekday - meeting_date.weekday()) % 7
        if mod and re.fullmatch(NEXT, mod):
            # "в следующий понедельник" = that weekday in the next calendar week.
            return meeting_date + timedelta(days=7 - meeting_date.weekday() + weekday)
        if mod and ahead == 0:
            return meeting_date  # "в эту среду" said on Wednesday means today.
        # Plain weekday: the nearest upcoming one; said on the same weekday it means next week.
        return meeting_date + timedelta(days=ahead or 7)
    if re.search(_word(r"послезавтра|бүрсігүні"), lowered):
        return meeting_date + timedelta(days=2)
    if re.search(_word(r"завтра\w*|ертең\w*"), lowered):
        return meeting_date + timedelta(days=1)
    if re.search(_word(r"сегодня|бүгін\w*"), lowered):
        return meeting_date
    if re.search(rf"(?:{NEXT})\s+(?:недел\w*|апта\w*)", lowered):
        # "на следующей неделе" is a range; the deadline is the end of that work week.
        return meeting_date + timedelta(days=7 - meeting_date.weekday() + 4)
    this_week = rf"(?:{THIS})\s+(?:недел\w*|апта\w*)|конц\w*\s+(?:{THIS}\s+)?недел\w*"
    if re.search(rf"{this_week}|апта\w*\s+соң\w*", lowered):
        friday = meeting_date + timedelta(days=4 - meeting_date.weekday())
        return friday if friday >= meeting_date else None
    if re.search(_word(r"через\s+неделю|бір\s+аптадан\s+кейін"), lowered):
        return meeting_date + timedelta(days=7)
    return None


DEADLINE_PHRASE = re.compile(
    r"(?:(?:в|во|до|к|ко|на|через)\s+)?(?:конц\w*\s+)?(?:(?:" + NEXT + "|" + THIS + r")\s+)?"
    r"(?:" + "|".join(WEEKDAYS.values()) + r"|послезавтра|бүрсігүні|завтра|ертең\w*"
    r"|недел\w*|апта\w*)(?:\s+дейін)?",
)


def find_deadline_phrase(text: str, meeting_date: date) -> str | None:
    """Find resolvable deadline wording inside a quote when the model omitted deadline_text."""
    lowered = text.casefold().replace("ё", "е")
    for match in DEADLINE_PHRASE.finditer(lowered):
        start, end = match.span()
        if (start and lowered[start - 1].isalnum()) or (
            end < len(lowered) and lowered[end].isalnum()
        ):
            continue
        if match.group(0).startswith("в среде"):
            continue  # "в среде" means "in the environment", not a Wednesday deadline.
        if resolve_deadline(match.group(0), meeting_date):
            return text[start:end]
    return None
