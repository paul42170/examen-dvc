"""
Script 4 - Entrainement du modele.

Charge les meilleurs hyperparametres trouves par le GridSearch
(models/best_params.pkl) et entraine un GradientBoostingRegressor final
sur l'ensemble d'entrainement normalise. Le modele entraine est sauvegarde
dans le dossier models (gbrt_model.pkl).
"""
import os
import yaml
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor

PROCESSED_DIR = os.path.join("data", "processed_data")
MODELS_DIR = "models"


def load_params(path: str = "params.yaml") -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def train_model() -> None:
    random_state = load_params().get("train", {}).get("random_state", 42)

    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train_scaled.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).values.ravel()

    best_params = joblib.load(os.path.join(MODELS_DIR, "best_params.pkl"))
    print(f"Entrainement avec les parametres : {best_params}")

    model = GradientBoostingRegressor(random_state=random_state, **best_params)
    model.fit(X_train, y_train)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(model, os.path.join(MODELS_DIR, "gbrt_model.pkl"))
    print(f"Modele entraine sauvegarde dans {MODELS_DIR}/gbrt_model.pkl")


if __name__ == "__main__":
    train_model()
