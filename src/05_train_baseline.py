"""
Cancer Variant Prioritization AI

Step 5:
- Train a baseline Logistic Regression model
- Evaluate pathogenic variant classification
- Save model metrics and predictions

Author:
Ilgın Gizem Türkmen
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clinvar_ml_features.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

METRICS_FILE = RESULTS_DIR / "baseline_metrics.json"
PREDICTIONS_FILE = RESULTS_DIR / "baseline_predictions.csv"
CONFUSION_MATRIX_FILE = FIGURES_DIR / "baseline_confusion_matrix.png"

RANDOM_STATE = 42
TEST_SIZE = 0.20


def load_data() -> pd.DataFrame:
    """Load the model-ready feature dataset."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Model-ready dataset was not found: {INPUT_FILE}"
        )

    return pd.read_csv(INPUT_FILE)


def prepare_features_and_target(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """
    Separate model features from the target variable.

    VariationID is an identifier, not a predictive biological feature.
    """
    target = data["IsPathogenic"].copy()

    features = data.drop(
        columns=[
            "VariationID",
            "IsPathogenic",
        ]
    )

    return features, target


def split_data(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """Create stratified training and test sets."""
    return train_test_split(
        features,
        target,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=target,
    )


def scale_features(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Standardize features using training-set statistics."""
    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    x_train_scaled_df = pd.DataFrame(
        x_train_scaled,
        columns=x_train.columns,
        index=x_train.index,
    )

    x_test_scaled_df = pd.DataFrame(
        x_test_scaled,
        columns=x_test.columns,
        index=x_test.index,
    )

    return x_train_scaled_df, x_test_scaled_df


def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> LogisticRegression:
    """Train a class-balanced Logistic Regression model."""
    model = LogisticRegression(
        max_iter=2_000,
        class_weight="balanced",
        random_state=RANDOM_STATE,
    )

    model.fit(x_train, y_train)

    return model


def evaluate_model(
    model: LogisticRegression,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate model performance."""
    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="binary",
        zero_division=0,
    )

    accuracy = float((predictions == y_test).mean())
    roc_auc = roc_auc_score(y_test, probabilities)

    metrics = {
        "accuracy": accuracy,
        "precision_pathogenic": float(precision),
        "recall_pathogenic": float(recall),
        "f1_pathogenic": float(f1),
        "roc_auc": float(roc_auc),
        "test_samples": int(len(y_test)),
    }

    prediction_table = pd.DataFrame(
        {
            "Actual": y_test.to_numpy(),
            "Predicted": predictions,
            "PathogenicProbability": probabilities,
        }
    )

    print("=" * 70)
    print("BASELINE LOGISTIC REGRESSION RESULTS")
    print("=" * 70)

    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1 score:  {f1:.4f}")
    print(f"ROC-AUC:   {roc_auc:.4f}")

    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=[
                "Not pathogenic",
                "Pathogenic",
            ],
            digits=4,
            zero_division=0,
        )
    )

    return metrics, prediction_table


def save_results(
    metrics: dict[str, float],
    predictions: pd.DataFrame,
) -> None:
    """Save model metrics and predictions."""
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with METRICS_FILE.open("w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=4)

    predictions.to_csv(
        PREDICTIONS_FILE,
        index=False,
    )

    print(f"\nMetrics saved to:\n{METRICS_FILE}")
    print(f"\nPredictions saved to:\n{PREDICTIONS_FILE}")


def create_confusion_matrix_figure(
    model: LogisticRegression,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> None:
    """Create and save the confusion matrix."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    predictions = model.predict(x_test)
    matrix = confusion_matrix(y_test, predictions)

    display = ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=[
            "Not pathogenic",
            "Pathogenic",
        ],
    )

    display.plot(values_format="d")
    plt.title("Baseline Logistic Regression Confusion Matrix")
    plt.tight_layout()
    plt.savefig(
        CONFUSION_MATRIX_FILE,
        dpi=300,
    )
    plt.close()

    print(
        f"\nConfusion matrix saved to:\n"
        f"{CONFUSION_MATRIX_FILE}"
    )


def main() -> None:
    print("Cancer Variant Prioritization AI")
    print("Baseline model training")

    data = load_data()

    features, target = prepare_features_and_target(data)

    x_train, x_test, y_train, y_test = split_data(
        features,
        target,
    )

    x_train_scaled, x_test_scaled = scale_features(
        x_train,
        x_test,
    )

    print(f"\nTraining samples: {len(x_train_scaled)}")
    print(f"Test samples: {len(x_test_scaled)}")
    print(f"Number of features: {x_train_scaled.shape[1]}")

    model = train_model(
        x_train_scaled,
        y_train,
    )

    metrics, predictions = evaluate_model(
        model,
        x_test_scaled,
        y_test,
    )

    save_results(
        metrics,
        predictions,
    )

    create_confusion_matrix_figure(
        model,
        x_test_scaled,
        y_test,
    )


if __name__ == "__main__":
    main()