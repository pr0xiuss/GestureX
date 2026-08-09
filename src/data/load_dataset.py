from pathlib import Path

import pandas as pd


DATASET_PATH = (
    Path(__file__).resolve().parents[2]
    / "data"
    / "raw"
    / "hand_gesture_landmarks"
    / "gesture_landmarks.csv"
)

GESTURE_CLASSES = [
    "open",
    "close",
    "point",
    "peace",
    "thumb",
    "rock",
]


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)

    df.columns = df.columns.str.strip()

    if "gesture_label" not in df.columns:
        raise ValueError("Dataset is missing the gesture_label column")

    df["gesture_label"] = df["gesture_label"].str.strip()

    df = df[df["gesture_label"].isin(GESTURE_CLASSES)].copy()

    return df


def validate_dataset(df: pd.DataFrame) -> None:
    expected_feature_count = 21 * 3
    expected_column_count = expected_feature_count + 1

    if df.shape[1] != expected_column_count:
        raise ValueError(
            f"Expected {expected_column_count} columns, "
            f"got {df.shape[1]}"
        )

    if df.isnull().any().any():
        raise ValueError("Dataset contains missing values")

    if not set(df["gesture_label"].unique()).issubset(GESTURE_CLASSES):
        raise ValueError("Dataset contains unexpected gesture classes")

    feature_columns = [
        column
        for column in df.columns
        if column != "gesture_label"
    ]

    if len(feature_columns) != expected_feature_count:
        raise ValueError(
            f"Expected {expected_feature_count} feature columns, "
            f"got {len(feature_columns)}"
        )


if __name__ == "__main__":
    dataset = load_dataset()

    validate_dataset(dataset)

    print("Dataset validation: PASSED")
    print("Shape:", dataset.shape)

    print("\nClasses:")
    print(dataset["gesture_label"].value_counts())

    print("\nMissing values:")
    print(dataset.isnull().sum().sum())

    print("\nFeature count:")
    print(len(dataset.columns) - 1)