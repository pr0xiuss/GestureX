from collections import Counter, deque


class PredictionSmoother:
    def __init__(self, window_size=5):
        if window_size <= 0:
            raise ValueError("window_size must be greater than 0")

        self.predictions = deque(maxlen=window_size)

    def update(self, prediction):
        self.predictions.append(prediction)

        counts = Counter(self.predictions)

        return counts.most_common(1)[0][0]

    def reset(self):
        self.predictions.clear()