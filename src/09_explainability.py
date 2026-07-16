"""
Cancer Variant Prioritization AI

Step 9:
- Train the final XGBoost model
- Explain model predictions using SHAP
- Save global feature-importance visualizations

Author:
Ilgın Gizem Türkmen
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import shap
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clinvar_ml_features.csv"
)

FIGURES_DIR = PROJECT_ROOT / "figures"

SHAP_SUMMARY_FILE = FIGURES_DIR / "shap_summary.png"
SHAP_BAR_FILE = FIGURES_DIR / "shap_feature_importance.png"

RANDOM_STATE = 42
TEST_SIZE = 0.20
SHAP_SAMPLE_SIZE = 2_000


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
    """Separate predictive features and target."""
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
    """Calculate class-imbalance weight for XGBoost."""
    negative_count = int((y_train == 0).sum())
    positive_count = int((y_train == 1).sum())

    if positive_count == 0:
        raise ValueError("Training data contains no pathogenic samples.")

    return negative_count / positive_count


def train_model(
    x_train: pd.DataFrame,
    y_train: pd.Series,
) -> XGBClassifier:
    """Train the final XGBoost model."""
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
        scale_pos_weight=calculate_scale_pos_weight(y_train),
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(x_train, y_train)

    return model


def create_shap_figures(
    model: XGBClassifier,
    x_test: pd.DataFrame,
) -> None:
    """Create global SHAP explanation figures."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    sample_size = min(SHAP_SAMPLE_SIZE, len(x_test))

    x_sample = x_test.sample(
        n=sample_size,
        random_state=RANDOM_STATE,
    )

    print(f"Explaining {sample_size} test variants with SHAP...")

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(x_sample)

    shap.summary_plot(
        shap_values,
        x_sample,
        show=False,
        max_display=15,
    )
    plt.tight_layout()
    plt.savefig(
        SHAP_SUMMARY_FILE,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    shap.summary_plot(
        shap_values,
        x_sample,
        plot_type="bar",
        show=False,
        max_display=15,
    )
    plt.tight_layout()
    plt.savefig(
        SHAP_BAR_FILE,
        dpi=300,
        bbox_inches="tight",
    )
    plt.close()

    print(f"\nSHAP summary saved to:\n{SHAP_SUMMARY_FILE}")
    print(f"\nSHAP importance saved to:\n{SHAP_BAR_FILE}")


def main() -> None:
    print("Cancer Variant Prioritization AI")
    print("Explainable AI with SHAP")

    data = load_data()

    features, target = prepare_features_and_target(data)

    x_train, x_test, y_train, _ = split_data(
        features,
        target,
    )

    print(f"\nTraining samples: {len(x_train)}")
    print(f"SHAP test pool: {len(x_test)}")
    print(f"Number of features: {x_train.shape[1]}")

    model = train_model(
        x_train,
        y_train,
    )

    create_shap_figures(
        model,
        x_test,
    )


if __name__ == "__main__":
    main()