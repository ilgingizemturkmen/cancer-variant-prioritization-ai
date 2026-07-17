"""
Cancer Variant Prioritization AI

Phase 2 - Step 13:
- Load ClinVar + HGNC + IntOGen enriched data
- Create a binary pathogenicity target
- Generate biologically meaningful ML features
- Remove constant and leakage-prone columns
- Save the model-ready feature dataset

Author:
Ilgın Gizem Türkmen
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

INPUT_FILE = (
    PROCESSED_DIR
    / "clinvar_hgnc_intogen_enriched.csv"
)

OUTPUT_FILE = (
    PROCESSED_DIR
    / "clinvar_ml_features_v2.csv"
)

METADATA_FILE = (
    RESULTS_DIR
    / "feature_engineering_v2_metadata.json"
)


PATHOGENIC_TERMS = {
    "pathogenic",
    "likely pathogenic",
    "pathogenic/likely pathogenic",
}

BENIGN_TERMS = {
    "benign",
    "likely benign",
    "benign/likely benign",
}


def create_directories() -> None:
    """Create output directories when they do not exist."""
    PROCESSED_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    RESULTS_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


def load_data() -> pd.DataFrame:
    """Load the HGNC and IntOGen enriched ClinVar dataset."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            "Input dataset was not found:\n"
            f"{INPUT_FILE}"
        )

    data = pd.read_csv(
        INPUT_FILE,
        low_memory=False,
    )

    required_columns = [
        "PrimaryGene",
        "ClinicalSignificance",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required columns are missing: "
            f"{missing_columns}"
        )

    return data


def create_binary_target(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep clearly pathogenic and clearly benign variants.

    Target:
    1 = Pathogenic / Likely pathogenic
    0 = Benign / Likely benign
    """
    prepared = data.copy()

    significance = (
        prepared["ClinicalSignificance"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.lower()
    )

    prepared["Target"] = np.nan

    prepared.loc[
        significance.isin(PATHOGENIC_TERMS),
        "Target",
    ] = 1

    prepared.loc[
        significance.isin(BENIGN_TERMS),
        "Target",
    ] = 0

    prepared = prepared[
        prepared["Target"].notna()
    ].copy()

    prepared["Target"] = (
        prepared["Target"]
        .astype(int)
    )

    return prepared


def safe_numeric(
    data: pd.DataFrame,
    column: str,
    default: float = 0.0,
) -> pd.Series:
    """Convert a column to numeric safely."""
    if column not in data.columns:
        return pd.Series(
            default,
            index=data.index,
            dtype=float,
        )

    return (
        pd.to_numeric(
            data[column],
            errors="coerce",
        )
        .fillna(default)
    )


def create_intogen_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Create numerical features from IntOGen annotations."""
    featured = data.copy()

    q_value = safe_numeric(
        featured,
        "IntOGenMinQValue",
        default=1.0,
    )

    q_value = q_value.clip(
        lower=1e-300,
        upper=1.0,
    )

    featured["IntOGenEvidenceScore"] = (
        -np.log10(q_value)
    )

    featured["LogIntOGenDriverRecords"] = np.log1p(
        safe_numeric(
            featured,
            "IntOGenDriverRecords",
        )
    )

    featured["LogIntOGenCancerTypeCount"] = np.log1p(
        safe_numeric(
            featured,
            "IntOGenCancerTypeCount",
        )
    )

    featured["LogIntOGenCohortCount"] = np.log1p(
        safe_numeric(
            featured,
            "IntOGenCohortCount",
        )
    )

    featured["LogIntOGenTotalMutations"] = np.log1p(
        safe_numeric(
            featured,
            "IntOGenTotalMutations",
        )
    )

    featured["LogIntOGenMutatedSamples"] = np.log1p(
        safe_numeric(
            featured,
            "IntOGenTotalMutatedSamples",
        )
    )

    featured["IntOGenMutationBurdenRatio"] = (
        safe_numeric(
            featured,
            "IntOGenTotalMutations",
        )
        /
        (
            safe_numeric(
                featured,
                "IntOGenTotalMutatedSamples",
            )
            + 1
        )
    )

    featured["IntOGenCancerBreadth"] = (
        safe_numeric(
            featured,
            "IntOGenCancerTypeCount",
        )
        /
        (
            safe_numeric(
                featured,
                "IntOGenCohortCount",
            )
            + 1
        )
    )

    return featured


def create_hgnc_features(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """Create simple numerical features from HGNC annotations."""
    featured = data.copy()

    if "location" in featured.columns:
        location = (
            featured["location"]
            .fillna("")
            .astype(str)
            .str.strip()
        )

        featured["HasChromosomeLocation"] = (
            location.ne("")
            .astype(int)
        )

        featured["IsAutosomalGene"] = (
            location
            .str.extract(
                r"^(\d+)",
                expand=False,
            )
            .notna()
            .astype(int)
        )

        featured["IsXChromosomeGene"] = (
            location
            .str.upper()
            .str.startswith("X")
            .astype(int)
        )

        featured["IsYChromosomeGene"] = (
            location
            .str.upper()
            .str.startswith("Y")
            .astype(int)
        )

    else:
        featured["HasChromosomeLocation"] = 0
        featured["IsAutosomalGene"] = 0
        featured["IsXChromosomeGene"] = 0
        featured["IsYChromosomeGene"] = 0

    return featured


def create_review_status_score(
    data: pd.DataFrame,
) -> pd.DataFrame:
    """
    Convert ClinVar review status into an evidence-strength score.

    This score is not the pathogenicity label.
    It represents the review confidence level.
    """
    featured = data.copy()

    if "ReviewStatus" not in featured.columns:
        featured["ReviewEvidenceScore"] = 0
        return featured

    review_status = (
        featured["ReviewStatus"]
        .fillna("")
        .astype(str)
        .str.lower()
    )

    score = pd.Series(
        0,
        index=featured.index,
        dtype=int,
    )

    score.loc[
        review_status.str.contains(
            "practice guideline",
            na=False,
        )
    ] = 4

    score.loc[
        review_status.str.contains(
            "expert panel",
            na=False,
        )
    ] = 3

    score.loc[
        review_status.str.contains(
            "multiple submitters",
            na=False,
        )
    ] = 2

    score.loc[
        review_status.str.contains(
            "single submitter",
            na=False,
        )
    ] = 1

    featured["ReviewEvidenceScore"] = score

    return featured


def choose_feature_columns(
    data: pd.DataFrame,
) -> list[str]:
    """Select ML features that exist in the dataset."""
    candidate_features = [
        # Existing ClinVar-derived numerical features
        "PositionVCF",
        "Start",
        "Stop",
        "Length",
        "NumberSubmitters",
        "ReviewEvidenceScore",

        # HGNC features
        "HasGeneGroup",
        "HasAliasSymbol",
        "HasPreviousSymbol",
        "HasChromosomeLocation",
        "IsAutosomalGene",
        "IsXChromosomeGene",
        "IsYChromosomeGene",

        # IntOGen role features
        "IntOGenHasLoFRole",
        "IntOGenHasActivatingRole",
        "IntOGenHasAmbiguousRole",

        # IntOGen transformed numerical features
        "IntOGenEvidenceScore",
        "LogIntOGenDriverRecords",
        "LogIntOGenCancerTypeCount",
        "LogIntOGenCohortCount",
        "LogIntOGenTotalMutations",
        "LogIntOGenMutatedSamples",
        "IntOGenMutationBurdenRatio",
        "IntOGenCancerBreadth",
        "IntOGenMaxSampleFraction",
    ]

    return [
        column
        for column in candidate_features
        if column in data.columns
    ]


def convert_features_to_numeric(
    data: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """Convert selected features to numeric values."""
    converted = data.copy()

    for column in feature_columns:
        converted[column] = (
            pd.to_numeric(
                converted[column],
                errors="coerce",
            )
            .replace(
                [np.inf, -np.inf],
                np.nan,
            )
            .fillna(0)
        )

    return converted


def remove_constant_features(
    data: pd.DataFrame,
    feature_columns: list[str],
) -> tuple[list[str], list[str]]:
    """
    Remove columns with only one unique value.

    Constant columns cannot help a machine-learning model.
    """
    useful_features: list[str] = []
    removed_features: list[str] = []

    for column in feature_columns:
        unique_count = data[column].nunique(
            dropna=False
        )

        if unique_count <= 1:
            removed_features.append(column)
        else:
            useful_features.append(column)

    return useful_features, removed_features


def build_output_dataset(
    data: pd.DataFrame,
    feature_columns: list[str],
) -> pd.DataFrame:
    """
    Build the final model-ready table.

    PrimaryGene is preserved for grouped validation,
    but it will not automatically be used as an ML feature.
    """
    identifier_columns = [
        column
        for column in [
            "VariationID",
            "AlleleID",
            "PrimaryGene",
        ]
        if column in data.columns
    ]

    output_columns = (
        identifier_columns
        + feature_columns
        + ["Target"]
    )

    return data[
        output_columns
    ].copy()


def save_metadata(
    original_rows: int,
    final_data: pd.DataFrame,
    selected_features: list[str],
    removed_features: list[str],
) -> None:
    """Save feature-engineering information as JSON."""
    metadata = {
        "input_file": str(INPUT_FILE),
        "output_file": str(OUTPUT_FILE),
        "original_rows": original_rows,
        "final_rows": len(final_data),
        "target_distribution": {
            str(key): int(value)
            for key, value in (
                final_data["Target"]
                .value_counts()
                .sort_index()
                .items()
            )
        },
        "selected_feature_count": len(
            selected_features
        ),
        "selected_features": selected_features,
        "removed_constant_features": (
            removed_features
        ),
    }

    with METADATA_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(
            metadata,
            file,
            indent=2,
            ensure_ascii=False,
        )


def print_summary(
    original_rows: int,
    final_data: pd.DataFrame,
    selected_features: list[str],
    removed_features: list[str],
) -> None:
    """Print feature-engineering results."""
    print("=" * 75)
    print("FEATURE ENGINEERING V2 SUMMARY")
    print("=" * 75)

    print(f"Original rows: {original_rows}")
    print(f"Final labelled rows: {len(final_data)}")

    print("\nTarget distribution:")
    print(
        final_data["Target"]
        .value_counts()
        .sort_index()
        .rename(
            index={
                0: "Benign",
                1: "Pathogenic",
            }
        )
        .to_string()
    )

    print(
        f"\nSelected ML features: "
        f"{len(selected_features)}"
    )

    for feature in selected_features:
        print(f"  + {feature}")

    print(
        f"\nRemoved constant features: "
        f"{len(removed_features)}"
    )

    for feature in removed_features:
        print(f"  - {feature}")

    print("\nExample model-ready rows:")

    print(
        final_data
        .head(5)
        .to_string(index=False)
    )


def main() -> None:
    print("Cancer Variant Prioritization AI")
    print("Feature engineering v2")

    create_directories()

    raw_data = load_data()
    original_rows = len(raw_data)

    featured = create_binary_target(
        raw_data
    )

    featured = create_intogen_features(
        featured
    )

    featured = create_hgnc_features(
        featured
    )

    featured = create_review_status_score(
        featured
    )

    candidate_features = choose_feature_columns(
        featured
    )

    featured = convert_features_to_numeric(
        featured,
        candidate_features,
    )

    selected_features, removed_features = (
        remove_constant_features(
            featured,
            candidate_features,
        )
    )

    final_data = build_output_dataset(
        featured,
        selected_features,
    )

    final_data.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    save_metadata(
        original_rows,
        final_data,
        selected_features,
        removed_features,
    )

    print_summary(
        original_rows,
        final_data,
        selected_features,
        removed_features,
    )

    print(
        f"\nModel-ready dataset saved to:\n"
        f"{OUTPUT_FILE}"
    )

    print(
        f"\nMetadata saved to:\n"
        f"{METADATA_FILE}"
    )


if __name__ == "__main__":
    main()