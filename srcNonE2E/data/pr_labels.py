# Pitch label set: L-3..L8 and S-3..S8 (24 classes)

PITCH_CLASSES = [f"L{i}" for i in range(-3, 9)] + [f"S{i}" for i in range(-3, 9)]
PITCH_TO_ID = {p: i for i, p in enumerate(PITCH_CLASSES)}
ID_TO_PITCH = {i: p for p, i in PITCH_TO_ID.items()}
NUM_PITCH_CLASSES = len(PITCH_CLASSES)