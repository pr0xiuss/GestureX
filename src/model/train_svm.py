import joblib

from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC

from src.model.prepare_data import prepare_data


MODEL_PATH = "models/svm_gesture_classifier.joblib"


def train_svm():
    X_train, X_test, y_train, y_test = prepare_data()

    pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("classifier", SVC()),
        ]
    )

    parameter_grid = {
        "classifier__C": [0.1, 1, 10, 100],
        "classifier__gamma": ["scale", 0.01, 0.1, 1],
        "classifier__kernel": ["rbf"],
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

    print("SVM training completed.")
    print("Best parameters:", grid_search.best_params_)
    print("Best cross-validation F1:", grid_search.best_score_)
    print("Model saved to:", MODEL_PATH)


if __name__ == "__main__":
    train_svm()