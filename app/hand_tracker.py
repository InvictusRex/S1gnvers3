import cv2
import mediapipe as mp

class HandTracker:
    def __init__(self, max_num_hands=1):
        self.hands = mp.solutions.hands.Hands(static_image_mode=False,
                                              max_num_hands=max_num_hands,
                                              min_detection_confidence=0.7,
                                              min_tracking_confidence=0.7)
        self.mp_draw = mp.solutions.drawing_utils

    def get_landmarks(self, image):
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.hands.process(image_rgb)
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                return [(lm.x, lm.y, lm.z) for lm in hand_landmarks.landmark]
        return None