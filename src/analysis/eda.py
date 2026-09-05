from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from src.data.load_dataset import load_dataset


OUTPUT_DIR = Path("outputs/eda")


def create_output_directory():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def plot_gesture_distribution(df):
    class_counts = df["gesture_label"].value_counts()

    plt.figure(figsize=(9, 5))

    bars = plt.bar(
        class_counts.index,
        class_counts.values,
    )

    plt.title("Gesture Class Distribution")
    plt.xlabel("Gesture Class")
    plt.ylabel("Number of Samples")

    for bar, count in zip(bars, class_counts.values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            count + 0.5,
            str(count),
            ha="center",
            va="bottom",
        )

    plt.tight_layout()

    output_path = OUTPUT_DIR / "gesture_class_distribution.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_coordinate_distribution(df):
    feature_columns = [
        column
        for column in df.columns
        if column != "gesture_label"
    ]

    features = df[feature_columns].astype(float)

    x_values = features[
        [column for column in feature_columns if column.endswith("_x")]
    ].to_numpy().flatten()

    y_values = features[
        [column for column in feature_columns if column.endswith("_y")]
    ].to_numpy().flatten()

    z_values = features[
        [column for column in feature_columns if column.endswith("_z")]
    ].to_numpy().flatten()

    plt.figure(figsize=(9, 5))

    plt.hist(
        x_values,
        bins=30,
        alpha=0.6,
        label="X coordinates",
    )

    plt.hist(
        y_values,
        bins=30,
        alpha=0.6,
        label="Y coordinates",
    )

    plt.hist(
        z_values,
        bins=30,
        alpha=0.6,
        label="Z coordinates",
    )

    plt.title("Distribution of Hand Landmark Coordinates")
    plt.xlabel("Coordinate Value")
    plt.ylabel("Frequency")
    plt.legend()

    plt.tight_layout()

    output_path = OUTPUT_DIR / "landmark_coordinate_distribution.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def plot_index_fingertip_position_by_class(df):
    plt.figure(figsize=(9, 6))

    for gesture in sorted(df["gesture_label"].unique()):
        gesture_data = df[df["gesture_label"] == gesture]

        plt.scatter(
            gesture_data["landmark_8_x"],
            gesture_data["landmark_8_y"],
            label=gesture,
            alpha=0.7,
        )

    plt.title("Index Fingertip Position by Gesture")
    plt.xlabel("Index Fingertip X Coordinate")
    plt.ylabel("Index Fingertip Y Coordinate")
    plt.legend()

    plt.tight_layout()

    output_path = OUTPUT_DIR / "index_fingertip_position_by_gesture.png"
    plt.savefig(output_path, dpi=300)
    plt.close()

    print(f"Saved: {output_path}")


def print_eda_summary(df):
    feature_columns = [
        column
        for column in df.columns
        if column != "gesture_label"
    ]

    features = df[feature_columns].astype(float)

    print("\nEDA SUMMARY")
    print("===========")

    print("\nDataset:")
    print(f"Samples: {len(df)}")
    print(f"Numerical features: {len(feature_columns)}")

    print("\nGesture class distribution:")
    print(df["gesture_label"].value_counts())

    print("\nCoordinate ranges:")

    for coordinate in ["x", "y", "z"]:
        coordinate_columns = [
            column
            for column in feature_columns
            if column.endswith(f"_{coordinate}")
        ]

        values = features[coordinate_columns].to_numpy()

        print(
            f"{coordinate.upper()}: "
            f"min={values.min():.4f}, "
            f"max={values.max():.4f}, "
            f"mean={values.mean():.4f}"
        )

    print("\nFeature variance:")
    print(
        f"Minimum variance: "
        f"{features.var().min():.6f}"
    )

    print(
        f"Maximum variance: "
        f"{features.var().max():.6f}"
    )


def main():
    create_output_directory()

    df = load_dataset()

    print("Starting Exploratory Data Analysis...")

    plot_gesture_distribution(df)

    plot_coordinate_distribution(df)

    plot_index_fingertip_position_by_class(df)

    print_eda_summary(df)

    print("\nEDA completed successfully.")


if __name__ == "__main__":
    main()