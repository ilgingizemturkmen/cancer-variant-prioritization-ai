"""
Cancer Variant Prioritization AI

Phase 2 - Step 12:
- Load the IntOGen cancer driver compendium
- Aggregate cohort-level driver information to gene level
- Integrate IntOGen annotations with the HGNC-enriched ClinVar dataset
- Save the enriched dataset

Author:
Ilgın Gizem Türkmen
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

REFERENCE_DIR = PROJECT_ROOT / "data" / "reference" / "intogen"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

CLINVAR_HGNC_FILE = (
    PROCESSED_DIR
    / "clinvar_hgnc_enriched.csv"
)

OUTPUT_FILE = (
    PROCESSED_DIR
    / "clinvar_hgnc_intogen_enriched.csv"
)


REQUIRED_INTOGEN_COLUMNS = [
    "SYMBOL",
    "TRANSCRIPT",
    "COHORT",
    "CANCER_TYPE",
    "METHODS",
    "MUTATIONS",
    "SAMPLES",
    "%_SAMPLES_COHORT",
    "QVALUE_COMBINATION",
    "ROLE",
    "IS_DRIVER",
    "TOTAL_SAMPLES",
]


def find_intogen_file() -> Path:
    """
    Find the IntOGen Compendium_Cancer_Genes.tsv file.

    The exact release folder name may change between releases,
    so the file is located dynamically.
    """
    matches = list(
        REFERENCE_DIR.rglob(
            "Compendium_Cancer_Genes.tsv"
        )
    )

    if not matches:
        raise FileNotFoundError(
            "Compendium_Cancer_Genes.tsv was not found under:\n"
            f"{REFERENCE_DIR}"
        )

    if len(matches) > 1:
        print(
            "Warning: Multiple IntOGen compendium files found."
        )
        print(
            "The first file will be used:"
        )

        for path in matches:
            print(path)

    return matches[0]


def load_intogen(
    intogen_file: Path,
) -> pd.DataFrame:
    """Load and validate the IntOGen driver compendium."""
    intogen = pd.read_csv(
        intogen_file,
        sep="\t",
        low_memory=False,
    )

    missing_columns = [
        column
        for column in REQUIRED_INTOGEN_COLUMNS
        if column not in intogen.columns
    ]

    if missing_columns:
        raise ValueError(
            "Required IntOGen columns are missing: "
            f"{missing_columns}"
        )

    return intogen[
        REQUIRED_INTOGEN_COLUMNS
    ].copy()


def clean_intogen(
    intogen: pd.DataFrame,
) -> pd.DataFrame:
    """Clean and standardize IntOGen annotations."""
    cleaned = intogen.copy()

    cleaned["SYMBOL"] = (
        cleaned["SYMBOL"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    cleaned["CANCER_TYPE"] = (
        cleaned["CANCER_TYPE"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    cleaned["ROLE"] = (
        cleaned["ROLE"]
        .fillna("Unknown")
        .astype(str)
        .str.strip()
    )

    cleaned["METHODS"] = (
        cleaned["METHODS"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    cleaned["IS_DRIVER"] = (
        cleaned["IS_DRIVER"]
        .astype(str)
        .str.strip()
        .str.lower()
        .eq("true")
    )

    numeric_columns = [
        "MUTATIONS",
        "SAMPLES",
        "%_SAMPLES_COHORT",
        "QVALUE_COMBINATION",
        "TOTAL_SAMPLES",
    ]

    for column in numeric_columns:
        cleaned[column] = pd.to_numeric(
            cleaned[column],
            errors="coerce",
        )

    cleaned = cleaned[
        cleaned["SYMBOL"].ne("")
    ].copy()

    cleaned = cleaned[
        cleaned["IS_DRIVER"]
    ].copy()

    return cleaned


def combine_unique_values(
    series: pd.Series,
) -> str:
    """
    Combine unique non-empty values into one pipe-separated string.
    """
    values = {
        str(value).strip()
        for value in series.dropna()
        if str(value).strip()
    }

    return "|".join(
        sorted(values)
    )


def aggregate_intogen_by_gene(
    intogen: pd.DataFrame,
) -> pd.DataFrame:
    """Aggregate cohort-level IntOGen records into one row per gene."""
    aggregated = (
        intogen
        .groupby(
            "SYMBOL",
            as_index=False,
        )
        .agg(
            IntOGenCancerTypes=(
                "CANCER_TYPE",
                combine_unique_values,
            ),
            IntOGenCohorts=(
                "COHORT",
                combine_unique_values,
            ),
            IntOGenMethods=(
                "METHODS",
                combine_unique_values,
            ),
            IntOGenRoles=(
                "ROLE",
                combine_unique_values,
            ),
            IntOGenTranscripts=(
                "TRANSCRIPT",
                combine_unique_values,
            ),
            IntOGenDriverRecords=(
                "IS_DRIVER",
                "size",
            ),
            IntOGenCancerTypeCount=(
                "CANCER_TYPE",
                "nunique",
            ),
            IntOGenCohortCount=(
                "COHORT",
                "nunique",
            ),
            IntOGenTotalMutations=(
                "MUTATIONS",
                "sum",
            ),
            IntOGenTotalMutatedSamples=(
                "SAMPLES",
                "sum",
            ),
            IntOGenMaxSampleFraction=(
                "%_SAMPLES_COHORT",
                "max",
            ),
            IntOGenMinQValue=(
                "QVALUE_COMBINATION",
                "min",
            ),
            IntOGenMaxCohortSize=(
                "TOTAL_SAMPLES",
                "max",
            ),
        )
    )

    aggregated["IsIntOGenDriver"] = 1

    aggregated["IntOGenHasLoFRole"] = (
        aggregated["IntOGenRoles"]
        .str.contains(
            r"(?:^|\|)LoF(?:$|\|)",
            regex=True,
            na=False,
        )
        .astype(int)
    )

    aggregated["IntOGenHasActivatingRole"] = (
        aggregated["IntOGenRoles"]
        .str.contains(
            r"(?:^|\|)Act(?:$|\|)",
            regex=True,
            na=False,
        )
        .astype(int)
    )

    aggregated["IntOGenHasAmbiguousRole"] = (
        aggregated["IntOGenRoles"]
        .str.contains(
            r"(?:^|\|)ambiguous(?:$|\|)",
            regex=True,
            na=False,
        )
        .astype(int)
    )

    return aggregated

def load_clinvar_hgnc() -> pd.DataFrame:
    """Load the HGNC-enriched ClinVar dataset."""
    if not CLINVAR_HGNC_FILE.exists():
        raise FileNotFoundError(
            "HGNC-enriched ClinVar dataset was not found:\n"
            f"{CLINVAR_HGNC_FILE}"
        )

    clinvar = pd.read_csv(
        CLINVAR_HGNC_FILE,
        low_memory=False,
    )

    if "PrimaryGene" not in clinvar.columns:
        raise ValueError(
            "PrimaryGene column is missing from "
            "the ClinVar-HGNC dataset."
        )

    clinvar["PrimaryGene"] = (
        clinvar["PrimaryGene"]
        .fillna("")
        .astype(str)
        .str.strip()
        .str.upper()
    )

    return clinvar


def integrate_intogen(
    clinvar: pd.DataFrame,
    intogen_gene_level: pd.DataFrame,
) -> pd.DataFrame:
    """Merge gene-level IntOGen annotations into ClinVar."""
    enriched = clinvar.merge(
        intogen_gene_level,
        how="left",
        left_on="PrimaryGene",
        right_on="SYMBOL",
        validate="many_to_one",
    )

    integer_fill_columns = [
        "IsIntOGenDriver",
        "IntOGenDriverRecords",
        "IntOGenCancerTypeCount",
        "IntOGenCohortCount",
        "IntOGenTotalMutations",
        "IntOGenTotalMutatedSamples",
        "IntOGenMaxCohortSize",
        "IntOGenHasLoFRole",
        "IntOGenHasActivatingRole",
        "IntOGenHasAmbiguousRole",
    ]

    for column in integer_fill_columns:
        enriched[column] = (
            enriched[column]
            .fillna(0)
            .astype(int)
        )

    text_fill_columns = [
        "IntOGenCancerTypes",
        "IntOGenCohorts",
        "IntOGenMethods",
        "IntOGenRoles",
        "IntOGenTranscripts",
    ]

    for column in text_fill_columns:
        enriched[column] = (
            enriched[column]
            .fillna("Not annotated")
        )

    return enriched


def print_summary(
    intogen_raw: pd.DataFrame,
    intogen_clean: pd.DataFrame,
    intogen_gene_level: pd.DataFrame,
    enriched: pd.DataFrame,
) -> None:
    """Print integration statistics."""
    driver_variant_count = int(
        enriched["IsIntOGenDriver"].sum()
    )

    annotation_rate = (
        driver_variant_count / len(enriched)
        if len(enriched) > 0
        else 0
    )

    unique_driver_genes_in_clinvar = (
        enriched.loc[
            enriched["IsIntOGenDriver"].eq(1),
            "PrimaryGene",
        ]
        .nunique()
    )

    print("=" * 75)
    print("INTOGEN INTEGRATION SUMMARY")
    print("=" * 75)

    print(
        f"Raw IntOGen cohort-level rows: "
        f"{len(intogen_raw)}"
    )

    print(
        f"Filtered driver rows: "
        f"{len(intogen_clean)}"
    )

    print(
        f"Unique IntOGen driver genes: "
        f"{len(intogen_gene_level)}"
    )

    print(
        f"ClinVar-HGNC variant rows: "
        f"{len(enriched)}"
    )

    print(
        f"Variants in IntOGen driver genes: "
        f"{driver_variant_count}"
    )

    print(
        f"Variant annotation rate: "
        f"{annotation_rate:.2%}"
    )

    print(
        f"Unique ClinVar genes found in IntOGen: "
        f"{unique_driver_genes_in_clinvar}"
    )

    print("\nIntOGen role distribution at gene level:")

    print(
        intogen_gene_level[
            "IntOGenRoles"
        ]
        .value_counts()
        .head(15)
        .to_string()
    )

    print("\nClinVar genes and IntOGen annotations:")

    gene_summary_columns = [
        "PrimaryGene",
        "IsIntOGenDriver",
        "IntOGenRoles",
        "IntOGenCancerTypeCount",
        "IntOGenCohortCount",
        "IntOGenTotalMutations",
        "IntOGenMinQValue",
    ]

    gene_summary = (
        enriched[
            gene_summary_columns
        ]
        .drop_duplicates(
            subset=["PrimaryGene"]
        )
        .sort_values(
            [
                "IsIntOGenDriver",
                "IntOGenCancerTypeCount",
            ],
            ascending=[
                False,
                False,
            ],
        )
    )

    print(
        gene_summary.to_string(
            index=False
        )
    )


def main() -> None:
    print("Cancer Variant Prioritization AI")
    print("IntOGen integration")

    intogen_file = find_intogen_file()

    print(
        f"Using IntOGen file:\n"
        f"{intogen_file}"
    )

    intogen_raw = load_intogen(
        intogen_file
    )

    intogen_clean = clean_intogen(
        intogen_raw
    )

    intogen_gene_level = (
        aggregate_intogen_by_gene(
            intogen_clean
        )
    )

    clinvar_hgnc = load_clinvar_hgnc()

    enriched = integrate_intogen(
        clinvar_hgnc,
        intogen_gene_level,
    )

    enriched.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print_summary(
        intogen_raw,
        intogen_clean,
        intogen_gene_level,
        enriched,
    )

    print(
        f"\nIntOGen-enriched dataset saved to:\n"
        f"{OUTPUT_FILE}"
    )


if __name__ == "__main__":
    main()