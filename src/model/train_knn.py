import joblib

from sklearn.model_selection import GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.model.prepare_data import prepare_data


MODEL_PATH = "models/knn_gesture_classifier.joblib"


def train_knn():
    X_train, X_test, y_train, y_test = prepare_data()

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", KNeighborsClassifier()),
        ]
    )

    parameter_grid = {
        "classifier__n_neighbors": [3, 5, 7, 9, 11],
        "classifier__weights": ["uniform", "distance"],
        "classifier__metric": ["euclidean", "manhattan"],
    }

    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=parameter_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=-1,
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    joblib.dump(best_model, MODEL_PATH)

    print("KNN training completed.")
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation F1:", grid_search.best_score_)
    print("Model saved to:", MODEL_PATH)


if __name__ == "__main__":
    train_knn()