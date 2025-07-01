# Guide pour obtenir les données

Ce projet nécessite plusieurs fichiers de données volumineux qui ne sont pas inclus dans le repository GitHub en raison de leur taille.

## Fichiers nécessaires

### 1. Données d'accidents (dossier `data/`)
- `20241031_accidents_Fr_19_22_AH_q.csv` (215MB)
- `X_train.csv` (100MB)
- `X_test.csv` (43MB)
- `y_train.csv` (3.9MB)
- `y_test.csv` (1.7MB)
- `departements-france.csv` (3.3KB) - *Inclus dans le repo*

### 2. Modèles Machine Learning (dossier `models/`)
- `20250129_catboost_best_model.pkl`
- `20250129_xgb_best_model.pkl`
- `20250304_XGBoost_bin_best_shap_values.pkl`
- `20250304_catboost_bin_best_shap_values.pkl`

### 3. Données géographiques (dossier `communes-20220101-shp/`)
- Fichiers shapefile des communes françaises

## Comment obtenir ces fichiers

### Option 1 : Téléchargement depuis une source externe
Contactez l'auteur du projet ou consultez la documentation fournie pour obtenir les liens de téléchargement.

### Option 2 : Génération des fichiers
- Les fichiers `X_train.csv`, `X_test.csv`, `y_train.csv`, `y_test.csv` peuvent être générés à partir du fichier principal d'accidents
- Les modèles `.pkl` peuvent être réentraînés en utilisant les notebooks Jupyter fournis

## Structure attendue
Assurez-vous que les fichiers sont placés dans la structure suivante :
```
Projet_Accidents/
├── data/
│   ├── 20241031_accidents_Fr_19_22_AH_q.csv
│   ├── X_train.csv
│   ├── X_test.csv
│   ├── y_train.csv
│   ├── y_test.csv
│   └── departements-france.csv
├── models/
│   ├── 20250129_catboost_best_model.pkl
│   ├── 20250129_xgb_best_model.pkl
│   ├── 20250304_XGBoost_bin_best_shap_values.pkl
│   └── 20250304_catboost_bin_best_shap_values.pkl
└── communes-20220101-shp/
    └── [fichiers shapefile]
```

## Vérification
Pour vérifier que tous les fichiers sont présents, exécutez :
```bash
python check_data_files.py
```