import numpy as np

from src.data.load_dataset import load_dataset
from src.features.feature_pipeline import build_features


def main():
    df = load_dataset()

    features = build_features(df)

    print("Final feature shape:", features.shape)
    print("Contains NaN:", np.isnan(features).any())


if __name__ == "__main__":
    main()