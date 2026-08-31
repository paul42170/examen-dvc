"""
Script 2 - Normalisation des donnees.

Les variables du dataset sont sur des echelles tres differentes (debits,
pH, densite...). On ajuste un StandardScaler sur X_train uniquement (pour
eviter toute fuite d'information depuis le test set) puis on l'applique
a X_train et X_test. Les versions normalisees sont sauvegardees dans
data/processed_data, et le scaler ajuste est sauvegarde dans models/
pour pouvoir etre reutilise (ex. en inference).
"""
import os
import joblib
import pandas as pd
from sklearn.preprocessing import StandardScaler

PROCESSED_DIR = os.path.join("data", "processed_data")
MODELS_DIR = "models"


def normalize_data() -> None:
    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train.csv"))
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test.csv"))

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    X_train_scaled_df = pd.DataFrame(X_train_scaled, columns=X_train.columns)
    X_test_scaled_df = pd.DataFrame(X_test_scaled, columns=X_test.columns)

    X_train_scaled_df.to_csv(os.path.join(PROCESSED_DIR, "X_train_scaled.csv"), index=False)
    X_test_scaled_df.to_csv(os.path.join(PROCESSED_DIR, "X_test_scaled.csv"), index=False)

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(scaler, os.path.join(MODELS_DIR, "scaler.pkl"))

    print("Normalisation terminee (StandardScaler ajuste sur X_train).")
    print(f"Fichiers sauvegardes dans {PROCESSED_DIR}, scaler sauvegarde dans {MODELS_DIR}")


if __name__ == "__main__":
    normalize_data()
