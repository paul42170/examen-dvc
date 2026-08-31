"""
Script 1 - Split des donnees en ensembles d'entrainement et de test.

Lit data/raw_data/raw.csv, separe la variable cible `silica_concentrate`
(derniere colonne) des variables explicatives, retire la colonne `date`
(non utilisee comme feature numerique), puis cree un split train/test.
Les 4 datasets resultants (X_train, X_test, y_train, y_test) sont
sauvegardes dans data/processed_data.
"""
import os
import yaml
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DATA_PATH = os.path.join("data", "raw_data", "raw.csv")
OUTPUT_DIR = os.path.join("data", "processed_data")
TARGET_COL = "silica_concentrate"
DATE_COL = "date"


def load_params(path: str = "params.yaml") -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def split_data() -> None:
    params = load_params().get("split", {})
    test_size = params.get("test_size", 0.2)
    random_state = params.get("random_state", 42)

    df = pd.read_csv(RAW_DATA_PATH)

    columns_to_drop = [TARGET_COL]
    if DATE_COL in df.columns:
        columns_to_drop.append(DATE_COL)

    X = df.drop(columns=columns_to_drop)
    y = df[[TARGET_COL]]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    os.makedirs(OUTPUT_DIR, exist_ok=True)
    X_train.to_csv(os.path.join(OUTPUT_DIR, "X_train.csv"), index=False)
    X_test.to_csv(os.path.join(OUTPUT_DIR, "X_test.csv"), index=False)
    y_train.to_csv(os.path.join(OUTPUT_DIR, "y_train.csv"), index=False)
    y_test.to_csv(os.path.join(OUTPUT_DIR, "y_test.csv"), index=False)

    print(f"Split effectue : X_train={X_train.shape}, X_test={X_test.shape}")
    print(f"Fichiers sauvegardes dans {OUTPUT_DIR}")


if __name__ == "__main__":
    split_data()
