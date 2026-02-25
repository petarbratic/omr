"""
Pitch and duration label sets.
Pitch: L-3..L8, S-3..S8 (24 classes). Duration: quarter, eighth, ... (9 classes).
"""

# Pitch: L-3..L8 and S-3..S8 (24 classes)
PITCH_CLASSES = [f"L{i}" for i in range(-3, 9)] + [f"S{i}" for i in range(-3, 9)]
PITCH_TO_ID = {p: i for i, p in enumerate(PITCH_CLASSES)}
ID_TO_PITCH = {i: p for p, i in PITCH_TO_ID.items()}
NUM_PITCH_CLASSES = len(PITCH_CLASSES)

# Duration (9 classes)
DURATION_CLASSES = [
    "quarter",
    "eighth",
    "sixteenth",
    "thirty_second",
    "half",
    "whole",
    "double_whole",
    "quadruple_whole",
]
DURATION_TO_ID = {d: i for i, d in enumerate(DURATION_CLASSES)}
ID_TO_DURATION = {i: d for d, i in DURATION_TO_ID.items()}
NUM_DURATION_CLASSES = len(DURATION_CLASSES)
