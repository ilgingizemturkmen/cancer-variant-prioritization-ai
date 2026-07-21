from pathlib import Path

import pandas as pd
import requests


BASE_DIR = Path(__file__).resolve().parent.parent

INPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "clinvar_hgnc_intogen_gnomad_ensembl_enriched.csv"
)

OUTPUT_FILE = (
    BASE_DIR
    / "data"
    / "processed"
    / "clinvar_hgnc_intogen_gnomad_ensembl_civic_enriched.csv"
)

CIVIC_API = "https://civicdb.org/api/graphql"


def load_dataset() -> pd.DataFrame:
    df = pd.read_csv(INPUT_FILE)

    print(f"Loaded {len(df):,} variants.")

    return df


def fetch_civic_gene(symbol: str) -> dict | None:
    query = """
    query GetGenes($symbols: [String!]!) {
      genes(
        entrezSymbols: $symbols
        first: 1
      ) {
        nodes {
          id
          name
          entrezId
          description
        }
      }
    }
    """

    response = requests.post(
        CIVIC_API,
        headers={
            "Accept": "application/json",
            "Content-Type": "application/json",
        },
        json={
            "query": query,
            "variables": {
                "symbols": [symbol]
            },
        },
        timeout=30,
    )

    response.raise_for_status()

    result = response.json()

    if result.get("errors"):
        print("CIViC API error:")
        print(result["errors"])
        return None

    nodes = (
        result.get("data", {})
        .get("genes", {})
        .get("nodes", [])
    )

    if not nodes:
        print(f"{symbol} CIViC içinde bulunamadı.")
        return None

    return nodes[0]
def main() -> None:
    df = load_dataset()

    genes = (
        df["PrimaryGene"]
        .dropna()
        .astype(str)
        .str.strip()
        .loc[lambda values: values != ""]
        .unique()
    )

    print(f"Unique primary genes: {len(genes)}")

    civic_data = []

    for i, gene in enumerate(genes, start=1):
        print(f"[{i}/{len(genes)}] Querying {gene}")

        result = fetch_civic_gene(gene)

        if result is not None:
            civic_data.append(result)

    print()
    print(f"CIViC matched primary genes: {len(civic_data)}")

    civic_df = pd.DataFrame(civic_data)

    civic_df = civic_df.rename(
        columns={
            "id": "civic_id",
            "name": "civic_gene",
            "entrezId": "civic_entrez_id",
            "description": "civic_description",
        }
    )

    df = df.merge(
        civic_df,
        left_on="PrimaryGene",
        right_on="civic_gene",
        how="left",
    )

    print(df.head())
    print()
    print(f"Rows after merge: {len(df):,}")
    print(f"CIViC matched rows: {df['civic_id'].notna().sum():,}")
    print(f"Rows after merge: {len(df):,}")
    print(f"CIViC matched rows: {df['civic_id'].notna().sum():,}")

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"Saved: {OUTPUT_FILE}")

if __name__ == "__main__":
    main()
 