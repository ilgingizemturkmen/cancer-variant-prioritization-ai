"""
Cancer Variant Prioritization AI

Step 8:
- Train an XGBoost classifier
- Evaluate pathogenic variant classification
- Save metrics, predictions and feature importance

Author:
Ilgın Gizem Türkmen
"""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clinvar_ml_features.csv"
)

RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = PROJECT_ROOT / "figures"

METRICS_FILE = RESULTS_DIR / "xgboost_metrics.json"
PREDICTIONS_FILE = RESULTS_DIR / "xgboost_predictions.csv"
CONFUSION_MATRIX_FILE = FIGURES_DIR / "xgboost_confusion_matrix.png"
FEATURE_IMPORTANCE_FILE = FIGURES_DIR / "xgboost_feature_importance.png"

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
    """Separate predictive features and target variable."""
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


def calculate_scale_pos_weight(y_train: pd.Series) -> float:
    """
    Calculate the imbalance ratio.

    XGBoost uses this value to give more weight to the
    less frequent pathogenic class.
    """
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    if positive_count == 0:
        raise ValueError("Training data contains no pathogenic samples.")

    return negative_count / positive_count


def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> XGBClassifier:
    """Train the class-balanced XGBoost model."""
    scale_pos_weight = calculate_scale_pos_weight(y_train)

    print(f"scale_pos_weight: {scale_pos_weight:.4f}")

    model = XGBClassifier(
        n_estimators=500,
        learning_rate=0.05,
        max_depth=6,
        min_child_weight=5,
        subsample=0.85,
        colsample_bytree=0.85,
        reg_alpha=0.10,
        reg_lambda=1.0,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=scale_pos_weight,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(x_train, y_train)

    return model


def evaluate_model(
    model: XGBClassifier,
    x_test: pd.DataFrame,
    y_test: pd.Series,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Evaluate XGBoost performance."""
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
    print("XGBOOST RESULTS")
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
    """Save metrics and predictions."""
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
    model: XGBClassifier,
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
    plt.title("XGBoost Confusion Matrix")
    plt.tight_layout()
    plt.savefig(CONFUSION_MATRIX_FILE, dpi=300)
    plt.close()

    print(
        f"\nConfusion matrix saved to:\n"
        f"{CONFUSION_MATRIX_FILE}"
    )


def create_feature_importance_figure(
    model: XGBClassifier,
    feature_names: list[str],
) -> None:
    """Plot the most important XGBoost features."""
    importance_data = pd.DataFrame(
        {
            "Feature": feature_names,
            "Importance": model.feature_importances_,
        }
    )

    top_features = (
        importance_data
        .sort_values("Importance", ascending=False)
        .head(15)
        .sort_values("Importance")
    )

    plt.figure(figsize=(10, 7))
    plt.barh(
        top_features["Feature"],
        top_features["Importance"],
    )
    plt.title("XGBoost Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.tight_layout()
    plt.savefig(FEATURE_IMPORTANCE_FILE, dpi=300)
    plt.close()

    print(
        f"\nFeature importance figure saved to:\n"
        f"{FEATURE_IMPORTANCE_FILE}"
    )


def main() -> None:
    print("Cancer Variant Prioritization AI")
    print("XGBoost training")

    data = load_data()

    features, target = prepare_features_and_target(data)

    x_train, x_test, y_train, y_test = split_data(
        features,
        target,
    )

    print(f"\nTraining samples: {len(x_train)}")
    print(f"Test samples: {len(x_test)}")
    print(f"Number of features: {x_train.shape[1]}")

    model = train_model(
        x_train,
        y_train,
    )

    metrics, predictions = evaluate_model(
        model,
        x_test,
        y_test,
    )

    save_results(
        metrics,
        predictions,
    )

    create_confusion_matrix_figure(
        model,
        x_test,
        y_test,
    )

    create_feature_importance_figure(
        model,
        feature_names=x_train.columns.tolist(),
    )


if __name__ == "__main__":
    main()