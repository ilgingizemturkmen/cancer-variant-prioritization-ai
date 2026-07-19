from pathlib import Path
from urllib.parse import quote

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "clinvar_hgnc_intogen_gnomad_enriched.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "clinvar_hgnc_intogen_gnomad_ensembl_enriched.csv"
)

ENSEMBL_SERVER = "https://rest.ensembl.org"


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded {len(df):,} variants.")

    return df


def select_primary_gene(symbol: str) -> str:
    """
    Bir hücrede birden fazla gen varsa ilk geçerli gen sembolünü seçer.
    Örnek: DCTN5;PALB2 -> DCTN5
    """
    genes = [
        gene.strip()
        for gene in str(symbol).split(";")
        if gene.strip()
    ]

    if not genes:
        return ""

    return genes[0]


def fetch_ensembl_gene(symbol: str) -> dict | None:
    primary_gene = select_primary_gene(symbol)

    if not primary_gene:
        return None

    encoded_symbol = quote(primary_gene, safe="")

    url = (
        f"{ENSEMBL_SERVER}/lookup/symbol/"
        f"homo_sapiens/{encoded_symbol}"
    )

    try:
        response = requests.get(
            url,
            headers={
                "Accept": "application/json",
                "Content-Type": "application/json",
            },
            timeout=30,
        )

        if response.status_code in {400, 404}:
            print(f"Gene not found: {primary_gene}")
            return None

        response.raise_for_status()

        return response.json()

    except requests.RequestException as error:
        print(f"Ensembl request failed for {primary_gene}: {error}")
        return None


def create_ensembl_table(genes: list[str]) -> pd.DataFrame:
    records = []

    total_genes = len(genes)

    for index, original_symbol in enumerate(genes, start=1):
        primary_gene = select_primary_gene(original_symbol)

        print(
            f"[{index}/{total_genes}] "
            f"Fetching {original_symbol}"
        )

        result = fetch_ensembl_gene(original_symbol)

        record = {
            "GeneSymbol": original_symbol,
            "ensembl_primary_gene": primary_gene,
            "ensembl_gene_id": None,
            "ensembl_chromosome": None,
            "ensembl_start": None,
            "ensembl_end": None,
            "ensembl_strand": None,
            "ensembl_biotype": None,
            "ensembl_assembly": None,
        }

        if result is not None:
            record.update(
                {
                    "ensembl_gene_id": result.get("id"),
                    "ensembl_chromosome": result.get(
                        "seq_region_name"
                    ),
                    "ensembl_start": result.get("start"),
                    "ensembl_end": result.get("end"),
                    "ensembl_strand": result.get("strand"),
                    "ensembl_biotype": result.get("biotype"),
                    "ensembl_assembly": result.get(
                        "assembly_name"
                    ),
                }
            )

        records.append(record)

    return pd.DataFrame(records)


def main() -> None:
    df = load_dataset()

    gene_column = "GeneSymbol"

    if gene_column not in df.columns:
        raise KeyError(
            f"Required column not found: {gene_column}"
        )

    genes = (
        df[gene_column]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda series: series.ne("")]
        .drop_duplicates()
        .tolist()
    )

    print(f"Unique genes: {len(genes):,}")

    ensembl_df = create_ensembl_table(genes)

    existing_ensembl_columns = [
        column
        for column in ensembl_df.columns
        if column != gene_column
        and column in df.columns
    ]

    if existing_ensembl_columns:
        rename_map = {
            column: f"hgnc_{column}"
            for column in existing_ensembl_columns
        }

        df = df.rename(columns=rename_map)

    enriched_df = df.merge(
        ensembl_df,
        on=gene_column,
        how="left",
        validate="many_to_one",
    )

    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    enriched_df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    matched_rows = (
        enriched_df["ensembl_gene_id"]
        .notna()
        .sum()
    )

    unmatched_rows = len(enriched_df) - matched_rows

    match_rate = (
        matched_rows / len(enriched_df) * 100
        if len(enriched_df) > 0
        else 0
    )

    print("-" * 50)
    print(f"Output rows: {len(enriched_df):,}")
    print(f"Matched rows: {matched_rows:,}")
    print(f"Unmatched rows: {unmatched_rows:,}")
    print(f"Match rate: {match_rate:.2f}%")
    print(f"Saved: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()