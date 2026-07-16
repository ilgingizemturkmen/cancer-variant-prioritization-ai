"""
Cancer Variant Prioritization AI

Step 10:
- Query the official GDC API
- Download open-access TCGA-BRCA masked somatic mutation data
- Filter cancer-associated genes
- Compare TCGA-BRCA mutations with ClinVar variants
- Save integration results

Important:
ClinVar clinical significance and TCGA somatic mutation occurrence
represent different biological contexts. Their overlap is used here
for research-oriented annotation support, not clinical validation.

Author:
Ilgın Gizem Türkmen
"""

from __future__ import annotations

import gzip
import json
import subprocess
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]

RAW_DIR = PROJECT_ROOT / "data" / "raw"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"

CLINVAR_FILE = (
    PROCESSED_DIR
    / "clinvar_cancer_variants_clean.csv"
)

TCGA_METADATA_FILE = (
    RESULTS_DIR
    / "tcga_brca_file_metadata.json"
)

TCGA_MAF_FILE = (
    RAW_DIR
    / "tcga_brca_masked_somatic_mutation.maf.gz"
)

TCGA_FILTERED_FILE = (
    PROCESSED_DIR
    / "tcga_brca_cancer_gene_mutations.csv"
)

INTEGRATED_FILE = (
    PROCESSED_DIR
    / "clinvar_tcga_brca_integrated.csv"
)

INTEGRATION_SUMMARY_FILE = (
    RESULTS_DIR
    / "tcga_integration_summary.json"
)

GDC_FILES_ENDPOINT = "https://api.gdc.cancer.gov/files"
GDC_DATA_ENDPOINT = "https://api.gdc.cancer.gov/data"

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
    RESULTS_DIR.mkdir(parents=True, exist_ok=True)


def build_gdc_payload() -> dict:
    """Create the official GDC API query payload."""
    filters = {
        "op": "and",
        "content": [
            {
                "op": "in",
                "content": {
                    "field": "cases.project.project_id",
                    "value": ["TCGA-BRCA"],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "files.data_category",
                    "value": ["Simple Nucleotide Variation"],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "files.data_type",
                    "value": ["Masked Somatic Mutation"],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "files.data_format",
                    "value": ["MAF"],
                },
            },
            {
                "op": "in",
                "content": {
                    "field": "files.access",
                    "value": ["open"],
                },
            },
        ],
    }

    return {
        "filters": filters,
        "format": "JSON",
        "fields": (
            "file_id,file_name,file_size,md5sum,access,"
            "data_category,data_type,data_format,"
            "analysis.workflow_type,"
            "cases.project.project_id"
        ),
        "size": 100,
    }


def query_gdc_files() -> list[dict]:
    """
    Query the GDC API using curl.

    curl is used because it already worked reliably on this Mac,
    while Python urllib previously had an SSL certificate issue.
    """
    payload = build_gdc_payload()

    command = [
        "curl",
        "--silent",
        "--show-error",
        "--fail",
        "--request",
        "POST",
        "--header",
        "Content-Type: application/json",
        "--data-binary",
        json.dumps(payload),
        GDC_FILES_ENDPOINT,
    ]

    print("Querying the GDC API for TCGA-BRCA MAF files...")

    result = subprocess.run(
        command,
        check=True,
        capture_output=True,
        text=True,
    )

    response = json.loads(result.stdout)

    hits = (
        response
        .get("data", {})
        .get("hits", [])
    )

    if not hits:
        raise RuntimeError(
            "No open-access TCGA-BRCA masked somatic MAF files "
            "were returned by the GDC API."
        )

    RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    with TCGA_METADATA_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(hits, file, indent=4)

    print(f"GDC returned {len(hits)} matching file(s).")

    for index, hit in enumerate(hits, start=1):
        workflow = (
            hit.get("analysis", {})
            .get("workflow_type", "Unknown")
        )

        print(
            f"{index}. {hit.get('file_name')} | "
            f"{workflow} | "
            f"{hit.get('file_size')} bytes"
        )

    return hits


def select_tcga_file(hits: list[dict]) -> dict:
    """
    Prefer the ensemble masked somatic mutation workflow.

    If it is unavailable, select the largest returned MAF file.
    """
    preferred_workflow = (
        "Aliquot Ensemble Somatic Variant Merging and Masking"
    )

    preferred_hits = [
        hit
        for hit in hits
        if (
            hit.get("analysis", {})
            .get("workflow_type")
            == preferred_workflow
        )
    ]

    candidates = preferred_hits if preferred_hits else hits

    selected = max(
        candidates,
        key=lambda item: int(item.get("file_size", 0)),
    )

    print("\nSelected TCGA file:")
    print(f"File name: {selected['file_name']}")
    print(f"File ID: {selected['file_id']}")
    print(
        "Workflow: "
        f"{selected.get('analysis', {}).get('workflow_type')}"
    )

    return selected


def download_tcga_file(file_record: dict) -> None:
    """Download the selected open-access GDC file."""
    if TCGA_MAF_FILE.exists():
        print(
            "\nTCGA MAF file already exists:\n"
            f"{TCGA_MAF_FILE}"
        )
        return

    file_id = file_record["file_id"]
    download_url = f"{GDC_DATA_ENDPOINT}/{file_id}"

    print("\nDownloading TCGA-BRCA somatic mutation MAF...")

    command = [
        "curl",
        "--location",
        "--fail",
        "--retry",
        "3",
        "--output",
        str(TCGA_MAF_FILE),
        download_url,
    ]

    subprocess.run(
        command,
        check=True,
    )

    print(f"TCGA MAF downloaded to:\n{TCGA_MAF_FILE}")


def detect_compression(file_path: Path) -> str | None:
    """Detect whether the downloaded file is gzip-compressed."""
    with file_path.open("rb") as file:
        magic_bytes = file.read(2)

    if magic_bytes == b"\x1f\x8b":
        return "gzip"

    return None


def load_tcga_maf() -> pd.DataFrame:
    """Load the TCGA MAF while ignoring comment/header lines."""
    compression = detect_compression(TCGA_MAF_FILE)

    print("\nLoading TCGA-BRCA MAF data...")

    data = pd.read_csv(
        TCGA_MAF_FILE,
        sep="\t",
        comment="#",
        compression=compression,
        low_memory=False,
    )

    required_columns = {
        "Hugo_Symbol",
        "Chromosome",
        "Start_Position",
        "End_Position",
        "Reference_Allele",
        "Tumor_Seq_Allele2",
        "Variant_Classification",
        "Variant_Type",
        "Tumor_Sample_Barcode",
    }

    missing_columns = required_columns.difference(data.columns)

    if missing_columns:
        raise ValueError(
            "Required TCGA MAF columns are missing: "
            f"{sorted(missing_columns)}"
        )

    return data


def filter_tcga_genes(data: pd.DataFrame) -> pd.DataFrame:
    """Filter TCGA-BRCA mutations to the selected cancer genes."""
    filtered = data[
        data["Hugo_Symbol"].isin(CANCER_GENES)
    ].copy()

    selected_columns = [
        "Hugo_Symbol",
        "Chromosome",
        "Start_Position",
        "End_Position",
        "Reference_Allele",
        "Tumor_Seq_Allele2",
        "Variant_Classification",
        "Variant_Type",
        "Tumor_Sample_Barcode",
    ]

    filtered = filtered[selected_columns].copy()

    filtered["Chromosome"] = (
        filtered["Chromosome"]
        .astype(str)
        .str.replace("chr", "", regex=False)
    )

    filtered["Start_Position"] = pd.to_numeric(
        filtered["Start_Position"],
        errors="coerce",
    )

    filtered = filtered.dropna(
        subset=[
            "Chromosome",
            "Start_Position",
            "Reference_Allele",
            "Tumor_Seq_Allele2",
        ]
    )

    filtered["Start_Position"] = (
        filtered["Start_Position"]
        .astype(int)
    )

    filtered.to_csv(
        TCGA_FILTERED_FILE,
        index=False,
    )

    return filtered


def load_clinvar() -> pd.DataFrame:
    """Load the cleaned ClinVar cancer variant dataset."""
    if not CLINVAR_FILE.exists():
        raise FileNotFoundError(
            f"ClinVar file was not found: {CLINVAR_FILE}"
        )

    clinvar = pd.read_csv(
        CLINVAR_FILE,
        low_memory=False,
    )

    clinvar["Chromosome"] = (
        clinvar["Chromosome"]
        .astype(str)
        .str.replace("chr", "", regex=False)
    )

    clinvar["Start"] = pd.to_numeric(
        clinvar["Start"],
        errors="coerce",
    )

    clinvar = clinvar.dropna(
        subset=[
            "Chromosome",
            "Start",
            "ReferenceAllele",
            "AlternateAllele",
        ]
    )

    clinvar["Start"] = clinvar["Start"].astype(int)

    return clinvar


def integrate_clinvar_tcga(
    clinvar: pd.DataFrame,
    tcga: pd.DataFrame,
) -> pd.DataFrame:
    """
    Join ClinVar and TCGA by gene, chromosome, start, reference
    and alternate allele.

    This identifies TCGA-BRCA somatic mutations that also have
    a matching ClinVar record.
    """
    tcga_occurrence = (
        tcga.groupby(
            [
                "Hugo_Symbol",
                "Chromosome",
                "Start_Position",
                "Reference_Allele",
                "Tumor_Seq_Allele2",
            ],
            dropna=False,
        )
        .agg(
            TCGA_SampleCount=(
                "Tumor_Sample_Barcode",
                "nunique",
            ),
            TCGA_MutationCount=(
                "Tumor_Sample_Barcode",
                "size",
            ),
            TCGA_VariantClassifications=(
                "Variant_Classification",
                lambda values: ";".join(
                    sorted(set(map(str, values)))
                ),
            ),
            TCGA_VariantTypes=(
                "Variant_Type",
                lambda values: ";".join(
                    sorted(set(map(str, values)))
                ),
            ),
        )
        .reset_index()
    )

    integrated = clinvar.merge(
        tcga_occurrence,
        how="inner",
        left_on=[
            "PrimaryGene",
            "Chromosome",
            "Start",
            "ReferenceAllele",
            "AlternateAllele",
        ],
        right_on=[
            "Hugo_Symbol",
            "Chromosome",
            "Start_Position",
            "Reference_Allele",
            "Tumor_Seq_Allele2",
        ],
    )

    integrated = integrated.sort_values(
        [
            "TCGA_SampleCount",
            "NumberSubmitters",
        ],
        ascending=False,
    )

    integrated.to_csv(
        INTEGRATED_FILE,
        index=False,
    )

    return integrated


def save_summary(
    tcga: pd.DataFrame,
    integrated: pd.DataFrame,
) -> None:
    """Save integration statistics."""
    summary = {
        "tcga_project": "TCGA-BRCA",
        "selected_cancer_genes": sorted(CANCER_GENES),
        "tcga_filtered_mutation_rows": int(len(tcga)),
        "tcga_unique_samples": int(
            tcga["Tumor_Sample_Barcode"].nunique()
        ),
        "clinvar_tcga_exact_matches": int(len(integrated)),
        "matched_unique_variants": int(
            integrated["VariationID"].nunique()
            if not integrated.empty
            else 0
        ),
        "matched_genes": (
            sorted(
                integrated["PrimaryGene"]
                .dropna()
                .unique()
                .tolist()
            )
            if not integrated.empty
            else []
        ),
    }

    with INTEGRATION_SUMMARY_FILE.open(
        "w",
        encoding="utf-8",
    ) as file:
        json.dump(summary, file, indent=4)

    print("\n" + "=" * 70)
    print("TCGA-BRCA INTEGRATION SUMMARY")
    print("=" * 70)

    for key, value in summary.items():
        print(f"{key}: {value}")

    print(f"\nFiltered TCGA data saved to:\n{TCGA_FILTERED_FILE}")
    print(f"\nIntegrated data saved to:\n{INTEGRATED_FILE}")
    print(f"\nSummary saved to:\n{INTEGRATION_SUMMARY_FILE}")


def main() -> None:
    print("Cancer Variant Prioritization AI")
    print("TCGA-BRCA integration")

    create_directories()

    hits = query_gdc_files()
    selected_file = select_tcga_file(hits)

    download_tcga_file(selected_file)

    tcga_data = load_tcga_maf()
    tcga_filtered = filter_tcga_genes(tcga_data)

    clinvar_data = load_clinvar()

    integrated_data = integrate_clinvar_tcga(
        clinvar_data,
        tcga_filtered,
    )

    save_summary(
        tcga_filtered,
        integrated_data,
    )


if __name__ == "__main__":
    main()
    
    
    # Large genomic datasets
data/raw/*
!data/raw/.gitkeep

# Generated processed datasets
data/processed/*.csv
!data/processed/.gitkeep

# Generated model outputs
results/*.csv
results/*.json

# Generated figures
figures/*.png
figures/*.jpg
figures/*.jpeg