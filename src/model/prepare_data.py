from sklearn.model_selection import train_test_split

from src.data.load_dataset import load_dataset
from src.features.feature_pipeline import build_features


TEST_SIZE = 0.2
RANDOM_STATE = 42


def prepare_data():
    df = load_dataset()

    features = build_features(df)

    labels = df["gesture_label"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        features,
        labels,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )

    return X_train, X_test, y_train, y_test