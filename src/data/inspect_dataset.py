import numpy as np
import pandas as pd
from pandas.api.types import is_string_dtype

from src.data.load_dataset import (
    EXPECTED_LANDMARK_FEATURES,
    GESTURE_CLASSES,
    load_dataset,
)


def main():
    df = load_dataset()

    print("Dataset Shape:", df.shape)

    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 Rows:")
    print(df.head())

    print("\nData Types:")
    print(df.dtypes)

    print("\nMissing Values:")
    print(df.isnull().sum())

    print("\nTotal Missing Values:")
    print(df.isnull().sum().sum())

    print("\nDuplicate Records:")
    print(df.duplicated().sum())

    feature_columns = [
        column
        for column in df.columns
        if column != "gesture_label"
    ]

    numeric_features = df[feature_columns].astype(float)

    print("\nInfinite Values:")
    print(np.isinf(numeric_features.to_numpy()).sum())

    print("\nFeature Count:")
    print(len(feature_columns))

    print("\nExpected Feature Count:")
    print(EXPECTED_LANDMARK_FEATURES)

    print("\nGesture Classes:")
    print(df["gesture_label"].value_counts())

    print("\nExpected Gesture Classes:")
    print(GESTURE_CLASSES)

    print("\nBasic Statistics:")
    print(numeric_features.describe())

    print("\nCategorical Columns:")
    for column in df.columns:
        if is_string_dtype(df[column]):
            print(f"{column}: {df[column].nunique()} unique values")
            print(df[column].value_counts())


if __name__ == "__main__":
    main()