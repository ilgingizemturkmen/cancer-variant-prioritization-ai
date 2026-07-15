"""
Cancer Variant Prioritization AI

Step 2:
- Load the processed ClinVar cancer variant dataset
- Inspect class distribution
- Inspect missing values
- Inspect variant types
- Generate exploratory figures

Author:
Ilgın Gizem Türkmen
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_FILE = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "clinvar_cancer_variants.csv"
)

FIGURES_DIR = PROJECT_ROOT / "figures"


def load_data() -> pd.DataFrame:
    """Load the processed ClinVar cancer variant dataset."""
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Processed dataset was not found: {DATA_FILE}"
        )

    data = pd.read_csv(DATA_FILE, low_memory=False)

    return data


def print_dataset_summary(data: pd.DataFrame) -> None:
    """Print general information about the dataset."""
    print("=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(f"Rows: {data.shape[0]}")
    print(f"Columns: {data.shape[1]}")

    print("\nColumn names:")
    for column in data.columns:
        print(f"- {column}")

    print("\nMissing values:")
    missing_values = (
        data.isna()
        .sum()
        .sort_values(ascending=False)
    )

    print(missing_values.to_string())


def inspect_clinical_significance(data: pd.DataFrame) -> None:
    """Inspect the distribution of clinical significance labels."""
    print("\n" + "=" * 70)
    print("CLINICAL SIGNIFICANCE DISTRIBUTION")
    print("=" * 70)

    distribution = data["ClinicalSignificance"].value_counts()

    print(distribution.to_string())


def inspect_variant_types(data: pd.DataFrame) -> None:
    """Inspect the most frequent variant types."""
    print("\n" + "=" * 70)
    print("VARIANT TYPE DISTRIBUTION")
    print("=" * 70)

    distribution = data["Type"].value_counts()

    print(distribution.head(20).to_string())


def inspect_genes(data: pd.DataFrame) -> None:
    """Inspect cancer-gene frequencies."""
    print("\n" + "=" * 70)
    print("GENE DISTRIBUTION")
    print("=" * 70)

    distribution = data["GeneSymbol"].value_counts()

    print(distribution.head(30).to_string())


def create_clinical_significance_figure(
    data: pd.DataFrame,
) -> None:
    """Create a bar chart for the main clinical significance classes."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    main_classes = (
        data["ClinicalSignificance"]
        .value_counts()
        .head(8)
        .sort_values()
    )

    plt.figure(figsize=(10, 6))
    main_classes.plot(kind="barh")

    plt.title("Main ClinVar Clinical Significance Classes")
    plt.xlabel("Number of variants")
    plt.ylabel("Clinical significance")
    plt.tight_layout()

    output_file = (
        FIGURES_DIR
        / "clinical_significance_distribution.png"
    )

    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"\nFigure saved to:\n{output_file}")


def create_variant_type_figure(
    data: pd.DataFrame,
) -> None:
    """Create a bar chart for the main variant types."""
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)

    main_types = (
        data["Type"]
        .value_counts()
        .head(10)
        .sort_values()
    )

    plt.figure(figsize=(10, 6))
    main_types.plot(kind="barh")

    plt.title("Main ClinVar Variant Types")
    plt.xlabel("Number of variants")
    plt.ylabel("Variant type")
    plt.tight_layout()

    output_file = FIGURES_DIR / "variant_type_distribution.png"

    plt.savefig(output_file, dpi=300)
    plt.close()

    print(f"\nFigure saved to:\n{output_file}")


def main() -> None:
    print("Cancer Variant Prioritization AI")
    print("Exploratory data analysis")

    data = load_data()

    print_dataset_summary(data)
    inspect_clinical_significance(data)
    inspect_variant_types(data)
    inspect_genes(data)

    create_clinical_significance_figure(data)
    create_variant_type_figure(data)


if __name__ == "__main__":
    main()