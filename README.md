# Examen DVC et Dagshub
Dans ce dépôt vous trouverez l'architecture proposé pour mettre en place la solution de l'examen. 

```bash       
├── examen_dvc          
│   ├── data       
│   │   ├── processed      
│   │   └── raw       
│   ├── metrics       
│   ├── models      
│   │   ├── data      
│   │   └── models        
│   ├── src       
│   └── README.md.py       
```
N'hésitez pas à rajouter les dossiers ou les fichiers qui vous semblent pertinents.

Vous devez dans un premier temps *Fork* le repo et puis le cloner pour travailler dessus. Le rendu de cet examen sera le lien vers votre dépôt sur DagsHub. Faites attention à bien mettre https://dagshub.com/licence.pedago en tant que colaborateur avec des droits de lecture seulement pour que ce soit corrigé.

Vous pouvez télécharger les données à travers le lien suivant : https://datascientest-mlops.s3.eu-west-1.amazonaws.com/mlops_dvc_fr/raw.csv.

## Pipeline mis en place

6 scripts (5 demandés + 1 script d'import des données brutes) :

- `src/data/import_raw_data.py` : télécharge le dataset brut dans `data/raw_data/raw.csv`
- `src/data/split_data.py` : split train/test (target = `silica_concentrate`) → `data/processed_data`
- `src/data/normalize_data.py` : normalisation (StandardScaler) → `data/processed_data` + `models/scaler.pkl`
- `src/models/grid_search.py` : GridSearchCV sur un GradientBoostingRegressor → `models/best_params.pkl`
- `src/models/train_model.py` : entraînement du modèle final → `models/gbrt_model.pkl`
- `src/models/evaluate_model.py` : prédictions (`data/predictions.csv`) et métriques (`metrics/scores.json`)

Les hyperparamètres (test_size, grille du GridSearch, etc.) sont centralisés dans `params.yaml`.

### Reproduire le pipeline

```bash
pip install -r requirements.txt
dvc repro
```

### Remote DVC (DagsHub)

```bash
dvc remote add origin https://dagshub.com/paul42170/examen-dvc.dvc
dvc remote modify origin --local auth basic
dvc remote modify origin --local user <votre_user_dagshub>
dvc remote modify origin --local password <votre_token_dagshub>
dvc push
```
