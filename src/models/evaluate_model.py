"""
Script 5 - Evaluation du modele.

Charge le modele entraine (models/gbrt_model.pkl) et l'evalue sur
l'ensemble de test normalise. Sauvegarde :
  - les predictions dans data/predictions.csv
  - les metriques d'evaluation (mse, rmse, mae, r2) dans metrics/scores.json
"""
import os
import json
import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

PROCESSED_DIR = os.path.join("data", "processed_data")
MODELS_DIR = "models"
METRICS_DIR = "metrics"
DATA_DIR = "data"


def evaluate_model() -> None:
    X_test = pd.read_csv(os.path.join(PROCESSED_DIR, "X_test_scaled.csv"))
    y_test = pd.read_csv(os.path.join(PROCESSED_DIR, "y_test.csv")).values.ravel()

    model = joblib.load(os.path.join(MODELS_DIR, "gbrt_model.pkl"))

    predictions = model.predict(X_test)

    mse = mean_squared_error(y_test, predictions)
    rmse = float(np.sqrt(mse))
    mae = mean_absolute_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    scores = {
        "mse": mse,
        "rmse": rmse,
        "mae": mae,
        "r2": r2,
    }

    os.makedirs(METRICS_DIR, exist_ok=True)
    with open(os.path.join(METRICS_DIR, "scores.json"), "w") as f:
        json.dump(scores, f, indent=4)

    predictions_df = pd.DataFrame(
        {
            "silica_concentrate_true": y_test,
            "silica_concentrate_pred": predictions,
        }
    )
    predictions_df.to_csv(os.path.join(DATA_DIR, "predictions.csv"), index=False)

    print(f"Scores : {scores}")
    print(f"Predictions sauvegardees dans {DATA_DIR}/predictions.csv")
    print(f"Metriques sauvegardees dans {METRICS_DIR}/scores.json")


if __name__ == "__main__":
    evaluate_model()
