import streamlit as st
import cv2
from app.hand_tracker import HandTracker
from app.classifier import ASLClassifier
from app.utils import normalize_landmarks
import pyttsx3
import tempfile
import numpy as np

# Initialize
st.title("🤟 Real-Time ASL Detection & TTS")
tracker = HandTracker()
classifier = ASLClassifier("model/asl_classifier.pkl")
tts_engine = pyttsx3.init()
cap = cv2.VideoCapture(0)

buffer = ""
prev_pred = ""
stable_count = 0
stable_threshold = 10
placeholder = st.empty()

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    landmarks = tracker.get_landmarks(frame)
    if landmarks:
        landmarks = normalize_landmarks(landmarks)
        pred = classifier.predict(landmarks)

        if pred == prev_pred:
            stable_count += 1
        else:
            stable_count = 0
        prev_pred = pred

        if stable_count > stable_threshold:
            if pred == "SPACE":
                buffer += " "
            elif pred == "CLEAR":
                buffer = ""
            elif pred == "SPEAK":
                tts_engine.say(buffer)
                tts_engine.runAndWait()
            else:
                buffer += pred
            stable_count = 0

    # Render
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    placeholder.image(frame_rgb, channels="RGB")
    st.markdown(f"### Output: `{buffer}`")

cap.release()