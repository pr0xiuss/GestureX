import numpy as np

from src.model.prepare_data import prepare_data


def main():
    X_train, X_test, y_train, y_test = prepare_data()

    print("Training feature shape:", X_train.shape)
    print("Testing feature shape:", X_test.shape)

    print("Training label shape:", y_train.shape)
    print("Testing label shape:", y_test.shape)

    print("Training classes:", np.unique(y_train))
    print("Testing classes:", np.unique(y_test))

    print("Training NaN:", np.isnan(X_train).any())
    print("Testing NaN:", np.isnan(X_test).any())

    print("Training infinite values:", np.isinf(X_train).any())
    print("Testing infinite values:", np.isinf(X_test).any())


if __name__ == "__main__":
    main()