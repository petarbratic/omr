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