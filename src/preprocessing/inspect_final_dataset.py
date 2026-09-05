import numpy as np

from src.data.load_dataset import load_dataset
from src.features.feature_pipeline import build_features


def main():
    print("FINAL PRE-PROCESSED DATASET")
    print("===========================")

    # Load cleaned dataset
    df = load_dataset()

    print("\nOriginal cleaned dataset:")
    print(f"Samples: {len(df)}")
    print(f"Columns: {df.shape[1]}")

    # Build final ML features
    features = build_features(df)

    labels = df["gesture_label"].to_numpy()

    print("\nPre-processing pipeline:")
    print("1. Extract landmark coordinates")
    print("2. Convert to wrist-relative coordinates")
    print("3. Normalize hand scale")
    print("4. Calculate geometric distance features")
    print("5. Calculate angle features")

    print("\nFinal feature matrix:")
    print(f"Shape: {features.shape}")

    print("\nFeature count:")
    print(features.shape[1])

    print("\nTarget shape:")
    print(labels.shape)

    print("\nTarget classes:")
    print(np.unique(labels))

    print("\nMissing values:")
    print(np.isnan(features).sum())

    print("\nInfinite values:")
    print(np.isinf(features).sum())

    print("\nFinDocuments/sem5/AIML/project/al dataset validation:")

    if features.shape[0] != len(df):
        raise ValueError("Feature/sample count mismatch")

    if features.shape[1] != 82:
        raise ValueError(
            f"Expected 82 final features, got {features.shape[1]}"
        )

    if np.isnan(features).any():
        raise ValueError("Final features contain NaN values")

    if np.isinf(features).any():
        raise ValueError("Final features contain infinite values")

    print("PASSED")


if __name__ == "__main__":
    main()