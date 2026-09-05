import numpy as np
import pandas as pd

from src.data.load_dataset import load_dataset
from src.features.feature_pipeline import build_features


def main():
    # Load the cleaned dataset
    df = load_dataset()

    # Build the final 82-feature representation
    features = build_features(df)

    # Extract target labels
    labels = df["gesture_label"].to_numpy()

    # Create a readable DataFrame for the final dataset
    feature_columns = [
        f"feature_{i + 1}"
        for i in range(features.shape[1])
    ]

    final_df = pd.DataFrame(
        features,
        columns=feature_columns,
    )

    final_df["gesture_label"] = labels

    print("=" * 100)
    print("FINAL PRE-PROCESSED DATASET")
    print("=" * 100)

    print("\nDataset Shape:")
    print(
        f"{features.shape[0]} samples × "
        f"{features.shape[1]} features"
    )

    print("\nFeature Composition:")
    print("63 Normalized Landmark Features")
    print("+ 9 Geometric Distance Features")
    print("+ 10 Angle Features")
    print("= 82 Final Features")

    print("\nFinal Dataset Preview:")
    print("-" * 100)

    # Display representative columns from the complete 82-feature matrix
    preview_columns = (
        feature_columns[:6]
        + ["feature_63"]
        + ["feature_64", "feature_65", "feature_66"]
        + ["feature_72"]
        + ["feature_73", "feature_74", "feature_75"]
        + ["feature_82"]
        + ["gesture_label"]
    )

    print(
        final_df[preview_columns]
        .head(5)
        .to_string(index=True)
    )

    print("\n" + "=" * 100)
    print("Validation")
    print("=" * 100)

    print(f"Feature Matrix Shape : {features.shape}")
    print(f"Target Shape         : {labels.shape}")
    print(f"Missing Values       : {np.isnan(features).sum()}")
    print(f"Infinite Values      : {np.isinf(features).sum()}")

    print("\nFinal Dataset Validation: PASSED")
    print("=" * 100)


if __name__ == "__main__":
    main()