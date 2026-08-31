"""
Script 0 - Import des donnees brutes.

Telecharge le dataset de flottation (traitement du minerai / silice) depuis le
bucket S3 fourni par DataScientest et le sauvegarde dans data/raw_data/raw.csv.
"""
import os
import sys
import requests

DATA_URL = "https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv"
OUTPUT_PATH = os.path.join("data", "raw_data", "raw.csv")


def import_raw_data(url: str = DATA_URL, output_path: str = OUTPUT_PATH) -> None:
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    print(f"Telechargement des donnees depuis {url} ...")
    response = requests.get(url, timeout=60)
    response.raise_for_status()

    with open(output_path, "wb") as f:
        f.write(response.content)

    print(f"Donnees brutes sauvegardees dans {output_path}")


if __name__ == "__main__":
    try:
        import_raw_data()
    except requests.RequestException as exc:
        print(f"Erreur lors du telechargement des donnees : {exc}", file=sys.stderr)
        sys.exit(1)
