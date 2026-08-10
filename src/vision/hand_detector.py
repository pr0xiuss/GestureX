import cv2
import mediapipe as mp

from mediapipe.tasks import python
from mediapipe.tasks.python import vision


MODEL_PATH = "models/hand_landmarker.task"


class HandDetector:
    def __init__(self):
        base_options = python.BaseOptions(
            model_asset_path=MODEL_PATH
        )

        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )

        self.detector = vision.HandLandmarker.create_from_options(options)
        self.timestamp_ms = 0

    def detect(self, frame):
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB,)

        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame,)

        self.timestamp_ms += 33

        result = self.detector.detect_for_video(mp_image, self.timestamp_ms,)

        if not result.hand_landmarks:
            return None

        hand_landmarks = result.hand_landmarks[0]

        landmarks = [(landmark.x, landmark.y, landmark.z) for landmark in hand_landmarks]

        if len(landmarks) != 21:
            raise RuntimeError(f"Expected 21 landmarks, got {len(landmarks)}")

        handedness = (result.handedness[0][0].category_name if result.handedness else None)

        return {
            "landmarks": landmarks,
            "handedness": handedness,
        }

    def close(self):
        self.detector.close()


def main():
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        raise RuntimeError("Could not open webcam")

    detector = HandDetector()

    try:
        while True:
            success, frame = cap.read()

            if not success:
                break

            detection = detector.detect(frame)

            if detection:
                landmarks = detection["landmarks"]
                handedness = detection["handedness"]

                height, width, _ = frame.shape

                for x, y, _ in landmarks:
                    px = int(x * width)
                    py = int(y * height)

                    cv2.circle(
                        frame,
                        (px, py),
                        5,
                        (0, 255, 0),
                        -1,
                    )

                cv2.putText(
                    frame,
                    f"Hand: {handedness}",
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2,
                )

            cv2.imshow("GestureX - Hand Detection", frame,)

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()