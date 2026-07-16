"""
Mini Variant Classification Pipeline

Purpose:
- Demonstrate the complete machine learning workflow
- Use a small synthetic genomic dataset
- Reinforce data loading, preprocessing, feature engineering,
  model training and evaluation

Author:
Ilgın Gizem Türkmen
"""

from __future__ import annotations

import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


RANDOM_STATE = 42


def create_demo_data() -> pd.DataFrame:
    """Create a small synthetic genomic variant dataset."""
    data = pd.DataFrame(
        {
            "Gene": [
                "BRCA1",
                "BRCA2",
                "TP53",
                "ATM",
                "CHEK2",
                "PTEN",
                "PALB2",
                "PIK3CA",
                "BRCA1",
                "TP53",
                "ATM",
                "CHEK2",
                "PTEN",
                "BRCA2",
                "PALB2",
                "AKT1",
                "ERBB2",
                "BRCA1",
                "ATM",
                "TP53",
            ],
            "VariantType": [
                "SNV",
                "Deletion",
                "SNV",
                "SNV",
                "Insertion",
                "Deletion",
                "SNV",
                "SNV",
                "Deletion",
                "SNV",
                "Insertion",
                "SNV",
                "Deletion",
                "SNV",
                "Insertion",
                "SNV",
                "Duplication",
                "SNV",
                "Deletion",
                "SNV",
            ],
            "VariantLength": [
                1,
                5,
                1,
                1,
                2,
                8,
                1,
                1,
                12,
                1,
                3,
                1,
                10,
                1,
                4,
                1,
                6,
                1,
                7,
                1,
            ],
            "NumberSubmitters": [
                8,
                6,
                10,
                7,
                1,
                5,
                4,
                2,
                9,
                8,
                2,
                1,
                6,
                3,
                2,
                1,
                2,
                7,
                5,
                9,
            ],
            "HasExpertReview": [
                1,
                1,
                1,
                1,
                0,
                1,
                1,
                0,
                1,
                1,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
            ],
            "IsPathogenic": [
                1,
                1,
                1,
                1,
                0,
                1,
                1,
                0,
                1,
                1,
                0,
                0,
                1,
                0,
                0,
                0,
                0,
                1,
                1,
                1,
            ],
        }
    )

    return data


def explore_data(data: pd.DataFrame) -> None:
    """Print basic EDA information."""
    print("=" * 60)
    print("EDA")
    print("=" * 60)

    print(f"Rows: {data.shape[0]}")
    print(f"Columns: {data.shape[1]}")

    print("\nMissing values:")
    print(data.isna().sum().to_string())

    print("\nTarget distribution:")
    print(data["IsPathogenic"].value_counts().to_string())

    print("\nGene distribution:")
    print(data["Gene"].value_counts().to_string())


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean the demo dataset."""
    cleaned = data.copy()

    cleaned["Gene"] = cleaned["Gene"].str.strip().str.upper()
    cleaned["VariantType"] = cleaned["VariantType"].str.strip()

    cleaned = cleaned.drop_duplicates()

    return cleaned


def create_features(
    data: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.Series]:
    """Create model-ready features and target."""
    target = data["IsPathogenic"].copy()

    features = data.drop(
        columns=["IsPathogenic"]
    )

    features = pd.get_dummies(
        features,
        columns=["Gene", "VariantType"],
        dtype=int,
    )

    return features, target


def split_and_scale(
    features: pd.DataFrame,
    target: pd.Series,
) -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.Series,
    pd.Series,
]:
    """Split the dataset and scale numerical features."""
    x_train, x_test, y_train, y_test = train_test_split(
        features,
        target,
        test_size=0.30,
        random_state=RANDOM_STATE,
        stratify=target,
    )

    scaler = StandardScaler()

    x_train_scaled = scaler.fit_transform(x_train)
    x_test_scaled = scaler.transform(x_test)

    x_train_scaled = pd.DataFrame(
        x_train_scaled,
        columns=x_train.columns,
        index=x_train.index,
    )

    x_test_scaled = pd.DataFrame(
        x_test_scaled,
        columns=x_test.columns,
        index=x_test.index,
    )

    return x_train_scaled, x_test_scaled, y_train, y_test


def train_and_evaluate(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> None:
    """Train Logistic Regression and print performance metrics."""
    model = LogisticRegression(
        class_weight="balanced",
        max_iter=1_000,
        random_state=RANDOM_STATE,
    )

    model.fit(x_train, y_train)

    predictions = model.predict(x_test)
    probabilities = model.predict_proba(x_test)[:, 1]

    print("\n" + "=" * 60)
    print("MODEL RESULTS")
    print("=" * 60)

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

    roc_auc = roc_auc_score(
        y_test,
        probabilities,
    )

    print(f"ROC-AUC: {roc_auc:.4f}")


def main() -> None:
    print("Mini Variant Classification Pipeline")

    data = create_demo_data()

    explore_data(data)

    cleaned_data = preprocess_data(data)

    features, target = create_features(cleaned_data)

    x_train, x_test, y_train, y_test = split_and_scale(
        features,
        target,
    )

    print("\nTraining shape:", x_train.shape)
    print("Test shape:", x_test.shape)

    train_and_evaluate(
        x_train,
        x_test,
        y_train,
        y_test,
    )


if __name__ == "__main__":
    main()