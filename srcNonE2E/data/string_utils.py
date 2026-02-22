"""
Input: (token string):
  - "note.quarter-L4"
  - "note.half-S-2"
  - "note.beamedRight2-L3"
  - "note.beamedBoth1-S3"
Output:
  - pitch: "L4", "S-2"
  - duration: "quarter" | "half" | "whole" | "eighth" | "sixteenth" | "thirty_second"
"""

import re
from typing import Optional, Tuple

# note.<kind>-<pitch>
_NOTE_TOKEN_RE = re.compile(r"^note\.(.+?)-(.+)$")

_BEAM_DIGIT_TO_DURATION = {
    0: "quarter",
    1: "eighth",
    2: "sixteenth",
    3: "thirty_second",
}

_EXPLICIT_DURATIONS = {
    "whole": "whole",
    "half": "half",
    "quarter": "quarter",
    "eighth": "eighth",
    "sixteenth": "sixteenth",
    "thirty_second": "thirty_second",
}

def parse_note_token(token: str) -> Optional[Tuple[str, str]]:
    """
    Parses a note token string.

    Returns:
        (duration, pitch) or None.

    Examples:
        parse_note_token("note.quarter-L4") -> ("quarter", "L4")
        parse_note_token("note.beamedRight2-L3") -> ("sixteenth", "L3")
    """
    m = _NOTE_TOKEN_RE.match(token.strip())
    if not m:
        return None

    kind = m.group(1)   # "quarter" or "beamedRight2"
    pitch = m.group(2)  # "L4" or "S-2"

    if "beamed" in kind:
        md = re.search(r"(\d+)$", kind)
        if not md:
            return None
        digit = int(md.group(1))
        duration = _BEAM_DIGIT_TO_DURATION.get(digit)
        if duration is None:
            return None
        return duration, pitch

    # Explicit duration case
    duration = _EXPLICIT_DURATIONS.get(kind)
    if duration is None:
        return None

    return duration, pitch


def is_note_token(token: str) -> bool:
    # True if token starts with "note".
    return token.strip().startswith("note.")


def extract_duration_and_pitch_from_transcript(transcript: str) -> list[Tuple[str, str, str]]:
    # Extracts (token, duration, pitch) for all note tokens in the transcript.
    out: list[Tuple[str, str, str]] = []
    for tok in transcript.strip().split():
        if not is_note_token(tok):
            continue
        parsed = parse_note_token(tok)
        if parsed is None:
            continue
        duration, pitch = parsed
        out.append((tok, duration, pitch))
    return out
    