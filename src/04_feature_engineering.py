"""
Cancer Variant Prioritization AI

Step 4:
- Create machine-learning features
- Encode categorical genomic variables
- Prepare model-ready training data

Author:
Ilgın Gizem Türkmen
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

INPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clinvar_cancer_variants_clean.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clinvar_ml_features.csv"
)


def load_data() -> pd.DataFrame:
    """Load the cleaned ClinVar dataset."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Clean dataset was not found: {INPUT_FILE}"
        )

    return pd.read_csv(INPUT_FILE, low_memory=False)


def create_features(data: pd.DataFrame) -> pd.DataFrame:
    """Create numerical and categorical ML features."""
    features = data.copy()

    features["ReferenceAlleleLength"] = (
        features["ReferenceAllele"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    features["AlternateAlleleLength"] = (
        features["AlternateAllele"]
        .fillna("")
        .astype(str)
        .str.len()
    )

    features["AlleleLengthDifference"] = (
        features["AlternateAlleleLength"]
        - features["ReferenceAlleleLength"]
    )

    features["HasExpertReview"] = (
        features["ReviewStatus"]
        .str.contains(
            "expert panel|practice guideline",
            case=False,
            na=False,
        )
        .astype(int)
    )

    features["HasMultipleSubmitters"] = (
        features["NumberSubmitters"]
        .fillna(0)
        .ge(2)
        .astype(int)
    )

    model_columns = [
        "VariationID",
        "PrimaryGene",
        "Type",
        "Chromosome",
        "VariantLength",
        "IsSNV",
        "IsIndel",
        "ReferenceAlleleLength",
        "AlternateAlleleLength",
        "AlleleLengthDifference",
        "NumberSubmitters",
        "HasExpertReview",
        "HasMultipleSubmitters",
        "IsPathogenic",
    ]

    model_data = features[model_columns].copy()

    categorical_columns = [
        "PrimaryGene",
        "Type",
        "Chromosome",
    ]

    model_data = pd.get_dummies(
        model_data,
        columns=categorical_columns,
        prefix=categorical_columns,
        dtype=int,
    )

    return model_data


def print_summary(data: pd.DataFrame) -> None:
    """Print feature-engineering summary."""
    print("=" * 70)
    print("FEATURE ENGINEERING SUMMARY")
    print("=" * 70)

    print(f"Rows: {data.shape[0]}")
    print(f"Model features: {data.shape[1] - 2}")

    print("\nFirst 20 model columns:")
    for column in data.columns[:20]:
        print(f"- {column}")

    print("\nTarget distribution:")
    print(
        data["IsPathogenic"]
        .value_counts()
        .rename(index={0: "Not pathogenic", 1: "Pathogenic"})
        .to_string()
    )

    print("\nFeature data types:")
    print(data.dtypes.value_counts().to_string())


def main() -> None:
    print("Cancer Variant Prioritization AI")
    print("Feature engineering")

    data = load_data()
    model_data = create_features(data)

    model_data.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print_summary(model_data)

    print(f"\nModel-ready dataset saved to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()