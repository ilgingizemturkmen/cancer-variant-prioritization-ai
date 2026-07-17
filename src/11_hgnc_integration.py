"""
Cancer Variant Prioritization AI

Phase 2 - Step 11:
- Download the official HGNC complete dataset
- Standardize gene symbols
- Add HGNC biological annotations to the ClinVar dataset
- Save an HGNC-enriched variant dataset

Author:
Ilgın Gizem Türkmen
"""

from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REFERENCE_DIR = PROJECT_ROOT / "data" / "reference"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

HGNC_URL = (
    "https://storage.googleapis.com/public-download-files/"
    "hgnc/tsv/tsv/hgnc_complete_set.txt"
)

HGNC_RAW_FILE = REFERENCE_DIR / "hgnc_complete_set.txt"

CLINVAR_FILE = (
    PROCESSED_DIR
    / "clinvar_cancer_variants_clean.csv"
)

OUTPUT_FILE = (
    PROCESSED_DIR
    / "clinvar_hgnc_enriched.csv"
)


HGNC_COLUMNS = [
    "hgnc_id",
    "symbol",
    "name",
    "locus_group",
    "locus_type",
    "status",
    "location",
    "alias_symbol",
    "prev_symbol",
    "gene_group",
    "entrez_id",
    "ensembl_gene_id",
]


def create_directories() -> None:
    """Create required project directories."""
    REFERENCE_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def download_hgnc() -> None:
    """Download the official HGNC complete dataset."""
    if HGNC_RAW_FILE.exists():
        print(f"HGNC file already exists:\n{HGNC_RAW_FILE}")
        return

    print("Downloading the official HGNC complete dataset...")

    command = [
        "curl",
        "--location",
        "--fail",
        "--retry",
        "3",
        "--output",
        str(HGNC_RAW_FILE),
        HGNC_URL,
    ]

    subprocess.run(
        command,
        check=True,
    )

    print(f"HGNC dataset downloaded to:\n{HGNC_RAW_FILE}")


def load_hgnc() -> pd.DataFrame:
    """Load and validate the HGNC dataset."""
    hgnc = pd.read_csv(
        HGNC_RAW_FILE,
        sep="\t",
        low_memory=False,
    )

    missing_columns = [
        column
        for column in HGNC_COLUMNS
        if column not in hgnc.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required HGNC columns are missing: "
            f"{missing_columns}"
        )

    hgnc = hgnc[HGNC_COLUMNS].copy()

    return hgnc


def clean_hgnc(hgnc: pd.DataFrame) -> pd.DataFrame:
    """Clean and standardize HGNC gene annotations."""
    cleaned = hgnc.copy()

    cleaned["symbol"] = (
        cleaned["symbol"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    cleaned = cleaned[
        cleaned["symbol"].ne("")
    ].copy()

    cleaned = cleaned[
        cleaned["status"].eq("Approved")
    ].copy()

    cleaned["IsProteinCoding"] = (
        cleaned["locus_group"]
        .fillna("")
        .str.contains(
            "protein-coding",
            case=False,
            na=False,
        )
        .astype(int)
    )

    cleaned["HasGeneGroup"] = (
        cleaned["gene_group"]
        .notna()
        .astype(int)
    )

    cleaned["HasAliasSymbol"] = (
        cleaned["alias_symbol"]
        .notna()
        .astype(int)
    )

    cleaned["HasPreviousSymbol"] = (
        cleaned["prev_symbol"]
        .notna()
        .astype(int)
    )

    cleaned = cleaned.drop_duplicates(
        subset=["symbol"]
    )

    return cleaned


def load_clinvar() -> pd.DataFrame:
    """Load the cleaned ClinVar cancer variant dataset."""
    if not CLINVAR_FILE.exists():
        raise FileNotFoundError(
            f"ClinVar dataset was not found:\n{CLINVAR_FILE}"
        )

    clinvar = pd.read_csv(
        CLINVAR_FILE,
        low_memory=False,
    )

    clinvar["PrimaryGene"] = (
        clinvar["PrimaryGene"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    return clinvar


def integrate_hgnc(
    clinvar: pd.DataFrame,
    hgnc: pd.DataFrame,
) -> pd.DataFrame:
    """Add HGNC gene annotations to ClinVar variants."""
    enriched = clinvar.merge(
        hgnc,
        how="left",
        left_on="PrimaryGene",
        right_on="symbol",
        validate="many_to_one",
    )

    enriched["HasHGNCAnnotation"] = (
        enriched["hgnc_id"]
        .notna()
        .astype(int)
    )

    return enriched


def print_summary(
    hgnc: pd.DataFrame,
    enriched: pd.DataFrame,
) -> None:
    """Print HGNC integration statistics."""
    annotated_count = int(
        enriched["HasHGNCAnnotation"].sum()
    )

    annotation_rate = (
        annotated_count / len(enriched)
        if len(enriched) > 0
        else 0
    )

    print("=" * 70)
    print("HGNC INTEGRATION SUMMARY")
    print("=" * 70)

    print(f"Approved HGNC genes: {len(hgnc)}")
    print(f"ClinVar variant rows: {len(enriched)}")
    print(f"HGNC-annotated variants: {annotated_count}")
    print(f"Annotation rate: {annotation_rate:.2%}")

    print("\nLocus group distribution:")
    print(
        enriched["locus_group"]
        .value_counts(dropna=False)
        .head(10)
        .to_string()
    )

    print("\nLocus type distribution:")
    print(
        enriched["locus_type"]
        .value_counts(dropna=False)
        .head(15)
        .to_string()
    )

    print("\nExample enriched records:")
    example_columns = [
        "PrimaryGene",
        "ClinicalSignificance",
        "hgnc_id",
        "name",
        "locus_group",
        "locus_type",
        "location",
        "gene_group",
        "IsProteinCoding",
    ]

    print(
        enriched[example_columns]
        .head(10)
        .to_string(index=False)
    )


def main() -> None:
    print("Cancer Variant Prioritization AI")
    print("HGNC integration")

    create_directories()
    download_hgnc()

    hgnc_raw = load_hgnc()
    hgnc_clean = clean_hgnc(hgnc_raw)

    clinvar = load_clinvar()

    enriched = integrate_hgnc(
        clinvar,
        hgnc_clean,
    )

    enriched.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print_summary(
        hgnc_clean,
        enriched,
    )

    print(f"\nHGNC-enriched dataset saved to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()