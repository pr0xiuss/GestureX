import joblib

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV

from src.model.prepare_data import prepare_data


MODEL_PATH = "models/random_forest_gesture_classifier.joblib"


def train_random_forest():
    X_train, X_test, y_train, y_test = prepare_data()

    model = RandomForestClassifier(
        random_state=42,
        class_weight="balanced",
        n_jobs=1,
    )

    parameter_grid = {
        "n_estimators": [100, 200, 300],
        "max_depth": [None, 10, 20],
        "min_samples_split": [2, 5],
        "min_samples_leaf": [1, 2],
    }

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=parameter_grid,
        cv=5,
        scoring="f1_macro",
        n_jobs=1,
    )

    grid_search.fit(X_train, y_train)

    best_model = grid_search.best_estimator_

    joblib.dump(best_model, MODEL_PATH)

    print("Random Forest training completed.")
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation F1:", grid_search.best_score_)
    print("Model saved to:", MODEL_PATH)


if __name__ == "__main__":
    train_random_forest()