from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import RepeatedStratifiedKFold, cross_validate

from src.model.prepare_data import prepare_data


MODELS_DIR = Path("models")

MODEL_PATHS = {
    "SVM": MODELS_DIR / "svm_gesture_classifier.joblib",
    "Random Forest": MODELS_DIR / "random_forest_gesture_classifier.joblib",
    "KNN": MODELS_DIR / "knn_gesture_classifier.joblib",
    "Logistic Regression": (
        MODELS_DIR / "logistic_regression_gesture_classifier.joblib"
    ),
}

RANDOM_STATE = 42
N_SPLITS = 5
N_REPEATS = 5
TOTAL_CV_RUNS = N_SPLITS * N_REPEATS


def evaluate_test_set(model, X_test, y_test):
    """Evaluate a trained model on the untouched test set."""

    predictions = model.predict(X_test)

    return {
        "accuracy": accuracy_score(y_test, predictions),
        "precision": precision_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "recall": recall_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        ),
        "f1": f1_score(
            y_test,
            predictions,
            average="macro",
            zero_division=0,
        ),
    }


def evaluate_cross_validation(model, X_train, y_train):
    """Evaluate a model using repeated stratified cross-validation."""

    cv = RepeatedStratifiedKFold(
        n_splits=N_SPLITS,
        n_repeats=N_REPEATS,
        random_state=RANDOM_STATE,
    )

    scoring = {
        "accuracy": "accuracy",
        "precision": "precision_macro",
        "recall": "recall_macro",
        "f1": "f1_macro",
    }

    scores = cross_validate(
        model,
        X_train,
        y_train,
        cv=cv,
        scoring=scoring,
        n_jobs=-1,
    )

    f1_scores = scores["test_f1"]

    perfect_f1_runs = int(
        np.sum(np.isclose(f1_scores, 1.0))
    )

    return {
        "accuracy_mean": np.mean(scores["test_accuracy"]),
        "accuracy_std": np.std(scores["test_accuracy"]),
        "precision_mean": np.mean(scores["test_precision"]),
        "precision_std": np.std(scores["test_precision"]),
        "recall_mean": np.mean(scores["test_recall"]),
        "recall_std": np.std(scores["test_recall"]),
        "f1_mean": np.mean(f1_scores),
        "f1_std": np.std(f1_scores),
        "perfect_f1_runs": perfect_f1_runs,
    }


def main():
    print("=" * 90)
    print("GESTUREX MODEL EVALUATION")
    print("=" * 90)

    # ---------------------------------------------------------
    # Prepare the fixed train/test split
    # ---------------------------------------------------------

    X_train, X_test, y_train, y_test = prepare_data()

    print("\nDataset Split")
    print("-" * 90)
    print(f"Training samples : {X_train.shape[0]}")
    print(f"Testing samples  : {X_test.shape[0]}")
    print(f"Features         : {X_train.shape[1]}")

    # ---------------------------------------------------------
    # Load trained models
    # ---------------------------------------------------------

    models = {}

    for model_name, model_path in MODEL_PATHS.items():

        if not model_path.exists():
            raise FileNotFoundError(
                f"Model file not found: {model_path}"
            )

        models[model_name] = joblib.load(model_path)

    # ---------------------------------------------------------
    # Repeated Cross-Validation
    # ---------------------------------------------------------

    print("\n" + "=" * 90)
    print("REPEATED CROSS-VALIDATION")
    print("=" * 90)

    print(
        f"\nUsing {N_SPLITS}-fold Stratified CV "
        f"repeated {N_REPEATS} times "
        f"({TOTAL_CV_RUNS} evaluations per model)"
    )

    cv_results = {}

    for model_name, model in models.items():

        print(f"\nEvaluating {model_name}...")

        cv_results[model_name] = evaluate_cross_validation(
            model,
            X_train,
            y_train,
        )

    cv_table = pd.DataFrame(
        {
            model_name: {
                "Accuracy": metrics["accuracy_mean"],
                "Precision": metrics["precision_mean"],
                "Recall": metrics["recall_mean"],
                "Macro F1": metrics["f1_mean"],
                "F1 Std Dev": metrics["f1_std"],
                "Perfect F1 Runs": (
                    f"{metrics['perfect_f1_runs']}/{TOTAL_CV_RUNS}"
                ),
            }
            for model_name, metrics in cv_results.items()
        }
    ).T

    print("\nCross-Validation Results")
    print("-" * 90)
    print(
        cv_table.to_string(
            float_format=lambda value: f"{value:.4f}"
        )
    )

    # ---------------------------------------------------------
    # Final Hold-Out Test Evaluation
    # ---------------------------------------------------------

    print("\n" + "=" * 90)
    print("FINAL HOLD-OUT TEST EVALUATION")
    print("=" * 90)

    print(
        "\nAll models are evaluated on the same "
        "untouched 41-sample test set."
    )

    test_results = {}

    for model_name, model in models.items():

        test_results[model_name] = evaluate_test_set(
            model,
            X_test,
            y_test,
        )

    test_table = pd.DataFrame(
        {
            model_name: {
                "Accuracy": metrics["accuracy"],
                "Precision": metrics["precision"],
                "Recall": metrics["recall"],
                "Macro F1": metrics["f1"],
            }
            for model_name, metrics in test_results.items()
        }
    ).T

    print("\nTest Set Results")
    print("-" * 90)
    print(
        test_table.to_string(
            float_format=lambda value: f"{value:.4f}"
        )
    )

    # ---------------------------------------------------------
    # Model Ranking
    # ---------------------------------------------------------

    print("\n" + "=" * 90)
    print("MODEL RANKING")
    print("=" * 90)

    ranking = cv_table.sort_values(
        by="Macro F1",
        ascending=False,
    )

    ranking = ranking[
        [
            "Macro F1",
            "F1 Std Dev",
            "Perfect F1 Runs",
            "Accuracy",
            "Precision",
            "Recall",
        ]
    ]

    print(
        "\nRanked using mean Macro F1 from "
        "repeated stratified cross-validation."
    )

    print("-" * 90)

    for rank, (model_name, row) in enumerate(
        ranking.iterrows(),
        start=1,
    ):
        print(
            f"{rank}. {model_name:<22} "
            f"F1={row['Macro F1']:.4f} "
            f"(±{row['F1 Std Dev']:.4f}) "
            f"Perfect={row['Perfect F1 Runs']}"
        )

    # ---------------------------------------------------------
    # Final Model Selection
    # ---------------------------------------------------------

    best_model = ranking.index[0]

    best_cv_f1 = cv_results[best_model]["f1_mean"]
    best_cv_std = cv_results[best_model]["f1_std"]
    best_perfect_runs = cv_results[best_model]["perfect_f1_runs"]
    best_test_f1 = test_results[best_model]["f1"]

    print("\n" + "=" * 90)
    print("FINAL MODEL SELECTION")
    print("=" * 90)

    print(f"\nSelected Model: {best_model}")

    print(f"Mean CV Macro F1 : {best_cv_f1:.4f}")
    print(f"CV F1 Std Dev    : {best_cv_std:.4f}")
    print(
        f"Perfect CV Runs  : "
        f"{best_perfect_runs}/{TOTAL_CV_RUNS}"
    )
    print(f"Test Macro F1    : {best_test_f1:.4f}")

    print("\nSelection Criteria:")
    print(
        "1. Highest mean Macro F1 across repeated "
        "stratified cross-validation"
    )
    print(
        "2. Lower F1 variation indicates more consistent performance"
    )
    print(
        "3. Final performance confirmed on the untouched test set"
    )

    print("\n" + "=" * 90)


if __name__ == "__main__":
    main()