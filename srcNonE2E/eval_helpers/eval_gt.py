# I used ChatGPT and Cursor for the development of this project.
# Helpers for obtaining ground-truth tokens from transcript.

from typing import List

from srcNonE2E.data.string_utils import extract_duration_and_pitch_from_transcript


def gt_tokens_from_transcript(transcript: str) -> List[str]:
    # Parse transcript and return list of tokens in form 'note.{duration}-{pitch}'.
    notes = extract_duration_and_pitch_from_transcript(transcript)
    out: List[str] = []
    for _, duration, pitch in notes:
        if duration is None or pitch is None:
            continue
        out.append(f"note.{duration}-{pitch}")
    return out
