"""
Cancer Variant Prioritization AI

Step 1:
- Download the official ClinVar variant summary file
- Read the dataset in chunks
- Select GRCh38 variants from cancer-associated genes
- Save a manageable processed dataset

Author:
Ilgın Gizem Türkmen
"""

from __future__ import annotations

import re
import urllib.request
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CLINVAR_URL = (
    "https://ftp.ncbi.nlm.nih.gov/pub/clinvar/"
    "tab_delimited/variant_summary.txt.gz"
)

RAW_FILE = RAW_DIR / "variant_summary.txt.gz"
OUTPUT_FILE = PROCESSED_DIR / "clinvar_cancer_variants.csv"


CANCER_GENES = {
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
}


def create_directories() -> None:
    """Create required project directories."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)


def download_clinvar() -> None:
    """Download ClinVar data unless it already exists."""
    if RAW_FILE.exists():
        print(f"ClinVar file already exists: {RAW_FILE}")
        return

    print("Downloading official ClinVar variant summary...")
    print("The file is large; downloading may take several minutes.")

    urllib.request.urlretrieve(CLINVAR_URL, RAW_FILE)

    print(f"Download completed: {RAW_FILE}")


def load_and_filter_clinvar() -> pd.DataFrame:
    """
    Read the compressed ClinVar file in chunks and select
    clinically classified GRCh38 variants from cancer genes.
    """
    selected_chunks: list[pd.DataFrame] = []

    columns = [
        "#AlleleID",
        "Type",
        "Name",
        "GeneID",
        "GeneSymbol",
        "ClinicalSignificance",
        "ClinSigSimple",
        "LastEvaluated",
        "ReviewStatus",
        "NumberSubmitters",
        "PhenotypeList",
        "Origin",
        "Assembly",
        "Chromosome",
        "Start",
        "Stop",
        "ReferenceAllele",
        "AlternateAllele",
        "VariationID",
    ]

    print("Reading and filtering ClinVar data...")

    reader = pd.read_csv(
        RAW_FILE,
        sep="\t",
        compression="gzip",
        usecols=columns,
        chunksize=100_000,
        low_memory=False,
    )

    gene_pattern = (
        r"(?:^|;)"
        + r"(?:"
        + "|".join(re.escape(gene) for gene in sorted(CANCER_GENES))
        + r")"
        + r"(?:;|$)"
    )

    for chunk_number, chunk in enumerate(reader, start=1):
        chunk = chunk.rename(columns={"#AlleleID": "AlleleID"})

        if chunk_number == 1:
            print("\nAvailable columns:")
            print(chunk.columns.tolist())

            print("\nAssembly distribution in first chunk:")
            print(chunk["Assembly"].value_counts(dropna=False).head(10))

            print("\nExample gene symbols:")
            print(chunk["GeneSymbol"].dropna().head(20).tolist())

        gene_mask = (
            chunk["GeneSymbol"]
            .fillna("")
            .astype(str)
            .str.contains(gene_pattern, regex=True)
        )

        clinical_mask = (
            chunk["ClinicalSignificance"].notna()
            & (chunk["ClinicalSignificance"].astype(str).str.strip() != "-")
        )

        filtered = chunk[
            (chunk["Assembly"] == "GRCh38")
            & gene_mask
            & clinical_mask
        ].copy()

        if not filtered.empty:
            selected_chunks.append(filtered)

        print(
            f"Chunk {chunk_number}: "
            f"{len(filtered)} selected variants"
        )

    if not selected_chunks:
        raise RuntimeError(
            "No matching ClinVar variants were found. "
            "Check Assembly and GeneSymbol values printed above."
        )

    cancer_variants = pd.concat(
        selected_chunks,
        ignore_index=True,
    )

    cancer_variants = cancer_variants.drop_duplicates(
        subset=["VariationID", "Assembly"]
    )

    return cancer_variants


def summarize_data(data: pd.DataFrame) -> None:
    """Print basic dataset quality-control information."""
    print("\nDataset successfully prepared.")
    print(f"Rows: {data.shape[0]}")
    print(f"Columns: {data.shape[1]}")

    print("\nSelected genes:")
    print(
        data["GeneSymbol"]
        .value_counts()
        .sort_index()
        .to_string()
    )

    print("\nClinical significance distribution:")
    print(
        data["ClinicalSignificance"]
        .value_counts()
        .head(15)
        .to_string()
    )

    print("\nFirst five variants:")
    print(
        data[
            [
                "GeneSymbol",
                "Name",
                "ClinicalSignificance",
                "ReviewStatus",
                "Chromosome",
                "Start",
            ]
        ]
        .head()
        .to_string(index=False)
    )


def main() -> None:
    print("=" * 60)
    print("Cancer Variant Prioritization AI")
    print("ClinVar data preparation")
    print("=" * 60)

    create_directories()
    download_clinvar()

    cancer_variants = load_and_filter_clinvar()
    cancer_variants.to_csv(OUTPUT_FILE, index=False)

    summarize_data(cancer_variants)

    print(f"\nProcessed dataset saved to:\n{OUTPUT_FILE}")


if __name__ == "__main__":
    main()