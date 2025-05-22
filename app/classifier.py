import joblib
import numpy as np

class ASLClassifier:
    def __init__(self, model_path):
        self.model = joblib.load(model_path)

    def predict(self, landmarks):
        flat = np.array(landmarks).flatten().reshape(1, -1)
        return self.model.predict(flat)[0]