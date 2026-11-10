import cv2
import joblib
import pandas as pd

from src.features.feature_pipeline import build_features
from src.features.landmark_features import LANDMARK_COLUMNS

from src.vision.hand_detector import HandDetector

from src.prediction.smoother import PredictionSmoother
from src.prediction.predictor import GesturePredictor

from src.actions.action_mapper import GestureActionMapper
from src.actions.desktop_controller import DesktopController


MODEL_PATH = "models/random_forest_gesture_classifier.joblib"

CONFIDENCE_THRESHOLD = 0.70
SMOOTHING_WINDOW = 5


HAND_CONNECTIONS = [
    # Thumb
    (0, 1),
    (1, 2),
    (2, 3),
    (3, 4),

    # Index finger
    (0, 5),
    (5, 6),
    (6, 7),
    (7, 8),

    # Middle finger
    (5, 9),
    (9, 10),
    (10, 11),
    (11, 12),

    # Ring finger
    (9, 13),
    (13, 14),
    (14, 15),
    (15, 16),

    # Pinky
    (13, 17),
    (17, 18),
    (18, 19),
    (19, 20),

    # Palm
    (0, 17),
]


def landmarks_to_dataframe(landmarks):
    values = []

    for landmark in landmarks:
        values.extend(landmark)

    return pd.DataFrame(
        [values],
        columns=LANDMARK_COLUMNS,
    )


def draw_hand_landmarks(frame, landmarks):
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

    for first, second in HAND_CONNECTIONS:
        x1, y1, _ = landmarks[first]
        x2, y2, _ = landmarks[second]

        point1 = (
            int(x1 * width),
            int(y1 * height),
        )

        point2 = (
            int(x2 * width),
            int(y2 * height),
        )

        cv2.line(
            frame,
            point1,
            point2,
            (0, 255, 0),
            2,
        )


def draw_text(frame, text, position, color=(0, 255, 0)):
    cv2.putText(
        frame,
        text,
        position,
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
    )


def main():
    model = joblib.load(MODEL_PATH)

    predictor = GesturePredictor(
        model,
        confidence_threshold=CONFIDENCE_THRESHOLD,
    )

    detector = HandDetector()

    smoother = PredictionSmoother(
        window_size=SMOOTHING_WINDOW,
    )

    action_mapper = GestureActionMapper(
        repeat_interval_frames=12,
        release_frames=3,
    )

    desktop = DesktopController()

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        detector.close()
        raise RuntimeError("Could not open webcam")

    action_status = "Desktop control ready"
    display_text = "Initializing"

    try:
        while True:
            frame_success, frame = cap.read()

            if not frame_success:
                break

            detection = detector.detect(frame)

            if detection:
                landmarks = detection["landmarks"]

                landmark_df = landmarks_to_dataframe(landmarks)
                features = build_features(landmark_df)

                result = predictor.predict(features)

                if result["accepted"]:
                    raw_prediction = result["gesture"]

                    prediction = smoother.update(raw_prediction)

                    confidence = result["confidence"]

                    display_text = (
                        f"Gesture: {prediction} "
                        f"({confidence * 100:.1f}%)"
                    )

                    action = action_mapper.update(prediction)

                    success_action = True
                    action_executed = False

                    print(
                        f"Raw: {raw_prediction} | "
                        f"Stable: {prediction} | "
                        f"Action: {action}"
                    )


                    cursor_active = (
                        raw_prediction == "point"
                        and prediction == "point"
                    )

                    if cursor_active and action == "move_mouse":
                        fingertip_x, fingertip_y, _ = landmarks[8]

                        if (
                            0.0 <= fingertip_x <= 1.0
                            and 0.0 <= fingertip_y <= 1.0
                        ):
                            success_action = desktop.execute_action(
                                "move_mouse",
                                x=fingertip_x,
                                y=fingertip_y,
                            )

                            action_executed = True

                        else:
                            success_action = False
                            action_status = "Invalid cursor coordinates"

                    else:
                        desktop.execute_action(None)


                    if (
                        raw_prediction != "point"
                        and prediction != "point"
                        and action is not None
                        and action != "move_mouse"
                    ):
                        success_action = desktop.execute_action(action)
                        action_executed = True


                    if action_executed:
                        if success_action:
                            action_status = f"Action: {action}"

                        elif action in {"play", "pause"}:
                            action_status = "No compatible media player"

                        else:
                            action_status = f"Action unavailable: {action}"

                else:
                    smoother.reset()
                    action_mapper.reset()

                    desktop.execute_action(None)

                    display_text = (
                        f"Uncertain "
                        f"({result['confidence'] * 100:.1f}%)"
                    )

                draw_hand_landmarks(frame, landmarks)

            else:
                smoother.reset()
                action_mapper.reset()

                desktop.execute_action(None)

                display_text = "No hand detected"

            draw_text(
                frame,
                display_text,
                (20, 40),
            )

            draw_text(
                frame,
                action_status,
                (20, 75),
                (255, 255, 0),
            )

            cv2.imshow(
                "GestureX - Live Inference",
                frame,
            )

            if cv2.waitKey(1) & 0xFF == ord("q"):
                break

    finally:
        desktop.execute_action(None)
        cap.release()
        cv2.destroyAllWindows()
        detector.close()


if __name__ == "__main__":
    main()
