"""
Cancer Variant Prioritization AI

Step 3:
- Standardize ClinVar clinical significance labels
- Create a machine-learning target variable
- Standardize cancer gene symbols
- Remove unsupported or ambiguous records
- Save the cleaned dataset

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
    / "clinvar_cancer_variants.csv"
)

OUTPUT_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clinvar_cancer_variants_clean.csv"
)


CANCER_GENES = [
    "BRCA1",
    "BRCA2",
    "TP53",
    "PTEN",
    "PALB2",
    "CHEK2",
    "ATM",
    "PIK3CA",
    "ERBB2",
    "AKT1",
]


def load_data() -> pd.DataFrame:
    """Load the processed ClinVar dataset."""
    if not INPUT_FILE.exists():
        raise FileNotFoundError(
            f"Input file was not found: {INPUT_FILE}"
        )

    return pd.read_csv(INPUT_FILE, low_memory=False)


def standardize_gene_symbol(gene_symbol: str) -> str | None:
    """
    Return the primary cancer gene found in a semicolon-separated
    GeneSymbol field.
    """
    symbols = {
        symbol.strip()
        for symbol in str(gene_symbol).split(";")
    }

    for gene in CANCER_GENES:
        if gene in symbols:
            return gene

    return None


def map_clinical_significance(label: str) -> str | None:
    """
    Map ClinVar labels into four broad research categories.

    Categories:
    - pathogenic
    - benign
    - uncertain
    - conflicting
    """
    normalized = str(label).strip().lower()

    if normalized in {
        "pathogenic",
        "likely pathogenic",
        "pathogenic/likely pathogenic",
        "pathogenic/likely pathogenic/pathogenic, low penetrance",
        "likely pathogenic, low penetrance",
    }:
        return "pathogenic"

    if normalized in {
        "benign",
        "likely benign",
        "benign/likely benign",
    }:
        return "benign"

    if normalized in {
        "uncertain significance",
        "uncertain significance/vus-mid",
    }:
        return "uncertain"

    if "conflicting classifications" in normalized:
        return "conflicting"

    return None


def preprocess_data(data: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize the ClinVar dataset."""
    cleaned = data.copy()

    cleaned["PrimaryGene"] = (
        cleaned["GeneSymbol"]
        .apply(standardize_gene_symbol)
    )

    cleaned["TargetClass"] = (
        cleaned["ClinicalSignificance"]
        .apply(map_clinical_significance)
    )

    cleaned = cleaned[
        cleaned["PrimaryGene"].notna()
        & cleaned["TargetClass"].notna()
    ].copy()

    cleaned["VariantLength"] = (
        cleaned["Stop"] - cleaned["Start"] + 1
    )

    cleaned["IsSNV"] = (
        cleaned["Type"]
        .eq("single nucleotide variant")
        .astype(int)
    )

    cleaned["IsIndel"] = (
        cleaned["Type"]
        .isin(["Indel", "Insertion", "Deletion"])
        .astype(int)
    )

    cleaned["IsPathogenic"] = (
        cleaned["TargetClass"]
        .eq("pathogenic")
        .astype(int)
    )

    cleaned = cleaned.drop_duplicates(
        subset=["VariationID", "Assembly"]
    )

    return cleaned


def print_summary(data: pd.DataFrame) -> None:
    """Print preprocessing results."""
    print("=" * 70)
    print("PREPROCESSING SUMMARY")
    print("=" * 70)

    print(f"Rows after preprocessing: {data.shape[0]}")
    print(f"Columns after preprocessing: {data.shape[1]}")

    print("\nTarget class distribution:")
    print(
        data["TargetClass"]
        .value_counts()
        .to_string()
    )

    print("\nPrimary gene distribution:")
    print(
        data["PrimaryGene"]
        .value_counts()
        .to_string()
    )

    print("\nBinary pathogenic target:")
    print(
        data["IsPathogenic"]
        .value_counts()
        .rename(index={0: "Not pathogenic", 1: "Pathogenic"})
        .to_string()
    )


def main() -> None:
    print("Cancer Variant Prioritization AI")
    print("Data preprocessing")

    data = load_data()
    cleaned_data = preprocess_data(data)

    cleaned_data.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print_summary(cleaned_data)

    print(f"\nClean dataset saved to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()