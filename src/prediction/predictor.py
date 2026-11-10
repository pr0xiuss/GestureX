import numpy as np


class GesturePredictor:
    def __init__(self, model, confidence_threshold=0.70):
        self.model = model
        self.confidence_threshold = confidence_threshold

    def predict(self, features):
        probabilities = self.model.predict_proba(features)[0]

        best_index = int(np.argmax(probabilities))
        confidence = float(probabilities[best_index])

        class_names = self.model.classes_
        prediction = class_names[best_index]

        if confidence < self.confidence_threshold:
            return {
                "gesture": None,
                "confidence": confidence,
                "accepted": False,
            }

        return {
            "gesture": prediction,
            "confidence": confidence,
            "accepted": True,
        }