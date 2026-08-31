"""
Script 3 - GridSearch des meilleurs hyperparametres.

Modele retenu : GradientBoostingRegressor (regression a base d'arbres
boostes), pertinent ici car les relations entre parametres operationnels
et concentration de silice sont probablement non lineaires.

Une recherche par grille (GridSearchCV, validation croisee) est effectuee
sur X_train_scaled / y_train. Les meilleurs parametres trouves sont
sauvegardes en .pkl dans le dossier models, pour etre reutilises par le
script d'entrainement.
"""
import os
import yaml
import joblib
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import GridSearchCV

PROCESSED_DIR = os.path.join("data", "processed_data")
MODELS_DIR = "models"


def load_params(path: str = "params.yaml") -> dict:
    if os.path.exists(path):
        with open(path, "r") as f:
            return yaml.safe_load(f) or {}
    return {}


def run_grid_search() -> None:
    params = load_params().get("grid_search", {})
    param_grid = params.get(
        "param_grid",
        {
            "n_estimators": [100, 200],
            "max_depth": [2, 3, 4],
            "learning_rate": [0.01, 0.1],
            "min_samples_leaf": [1, 3],
        },
    )
    cv = params.get("cv", 5)
    scoring = params.get("scoring", "neg_mean_squared_error")
    n_jobs = params.get("n_jobs", -1)

    X_train = pd.read_csv(os.path.join(PROCESSED_DIR, "X_train_scaled.csv"))
    y_train = pd.read_csv(os.path.join(PROCESSED_DIR, "y_train.csv")).values.ravel()

    model = GradientBoostingRegressor(random_state=42)

    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        cv=cv,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=1,
    )
    grid_search.fit(X_train, y_train)

    print(f"Meilleurs parametres trouves : {grid_search.best_params_}")
    print(f"Meilleur score ({scoring}) : {grid_search.best_score_:.4f}")

    os.makedirs(MODELS_DIR, exist_ok=True)
    joblib.dump(grid_search.best_params_, os.path.join(MODELS_DIR, "best_params.pkl"))
    print(f"Meilleurs parametres sauvegardes dans {MODELS_DIR}/best_params.pkl")


if __name__ == "__main__":
    run_grid_search()
