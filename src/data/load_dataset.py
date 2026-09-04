from pathlib import Path

import numpy as np
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

EXPECTED_LANDMARK_FEATURES = 21 * 3
EXPECTED_COLUMN_COUNT = EXPECTED_LANDMARK_FEATURES + 1


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(DATASET_PATH)

    df.columns = df.columns.str.strip()

    if "gesture_label" not in df.columns:
        raise ValueError("Dataset is missing the gesture_label column")

    df["gesture_label"] = df["gesture_label"].str.strip()

    df = df[df["gesture_label"].isin(GESTURE_CLASSES)].copy()

    return df


def validate_dataset(df: pd.DataFrame) -> None:
    # Check dataset structure
    if df.shape[1] != EXPECTED_COLUMN_COUNT:
        raise ValueError(
            f"Expected {EXPECTED_COLUMN_COUNT} columns, "
            f"got {df.shape[1]}"
        )

    # Check required target column
    if "gesture_label" not in df.columns:
        raise ValueError("Dataset is missing the gesture_label column")

    # Check feature columns
    feature_columns = [
        column
        for column in df.columns
        if column != "gesture_label"
    ]

    if len(feature_columns) != EXPECTED_LANDMARK_FEATURES:
        raise ValueError(
            f"Expected {EXPECTED_LANDMARK_FEATURES} feature columns, "
            f"got {len(feature_columns)}"
        )

    # Check missing values
    if df.isnull().any().any():
        missing_count = int(df.isnull().sum().sum())
        raise ValueError(
            f"Dataset contains {missing_count} missing values"
        )

    # Check duplicate records
    duplicate_count = int(df.duplicated().sum())

    if duplicate_count > 0:
        raise ValueError(
            f"Dataset contains {duplicate_count} duplicate records"
        )

    # Check numerical feature values
    try:
        numeric_features = df[feature_columns].astype(float)
    except (TypeError, ValueError) as exc:
        raise ValueError(
            "Dataset contains non-numeric landmark feature values"
        ) from exc

    # Check infinite values
    if np.isinf(numeric_features.to_numpy()).any():
        raise ValueError(
            "Dataset contains infinite landmark feature values"
        )

    # Check expected gesture classes
    actual_classes = set(df["gesture_label"].unique())
    expected_classes = set(GESTURE_CLASSES)

    unexpected_classes = actual_classes - expected_classes

    if unexpected_classes:
        raise ValueError(
            f"Dataset contains unexpected gesture classes: "
            f"{sorted(unexpected_classes)}"
        )

    # Check that all expected classes are present
    missing_classes = expected_classes - actual_classes

    if missing_classes:
        raise ValueError(
            f"Dataset is missing expected gesture classes: "
            f"{sorted(missing_classes)}"
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

    print("\nDuplicate records:")
    print(dataset.duplicated().sum())

    print("\nInfinite values:")
    feature_columns = [
        column
        for column in dataset.columns
        if column != "gesture_label"
    ]

    numeric_features = dataset[feature_columns].astype(float)

    print(np.isinf(numeric_features.to_numpy()).sum())

    print("\nFeature count:")
    print(len(feature_columns))