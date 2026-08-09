import pandas as pd


DATASET_PATH = "data/raw/hand_gesture_landmarks/gesture_landmarks.csv"


def main():
    df = pd.read_csv(DATASET_PATH)

    print("Shape:", df.shape)
    print("\nColumns:")
    print(df.columns.tolist())

    print("\nFirst 5 rows:")
    print(df.head())

    print("\nData types:")
    print(df.dtypes)

    print("\nMissing values:")
    print(df.isnull().sum())

    print("\nBasic statistics:")
    print(df.describe())

    print("\nUnique values:")
    for column in df.columns:
        if df[column].dtype == "object":
            print(f"{column}: {df[column].nunique()} unique values")
            print(df[column].value_counts())


if __name__ == "__main__":
    main()