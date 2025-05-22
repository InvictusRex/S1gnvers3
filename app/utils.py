import numpy as np

def normalize_landmarks(landmarks):
    base_x, base_y, base_z = landmarks[0]
    return [(x - base_x, y - base_y, z - base_z) for x, y, z in landmarks]