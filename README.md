# Projet Accidents Routiers en France (2019-2022)

## Description
Cette application Streamlit permet d'explorer et d'analyser les données d'accidents routiers en France sur la période 2019-2022. Elle offre des visualisations interactives, des analyses statistiques et des modèles prédictifs pour comprendre les facteurs de risque associés aux accidents de la route.

## Fonctionnalités principales
- **Introduction** : Présentation du projet et de ses objectifs
- **Exploration** : Analyse exploratoire des données avec visualisations interactives
- **DataVisualisation** : Graphiques et cartes pour comprendre les tendances des accidents
- **Modélisation** : Modèles prédictifs (XGBoost et CatBoost) pour la gravité des accidents avec analyse SHAP
- **Conclusion** : Synthèse des résultats et perspectives

## Technologies utilisées
- **Framework** : Streamlit
- **Analyse de données** : Pandas, NumPy
- **Visualisation** : Matplotlib, Seaborn, Plotly
- **Cartographie** : GeoPandas, Fiona, PyProj
- **Machine Learning** : Scikit-learn, XGBoost, CatBoost
- **Interprétabilité** : SHAP

## Installation

### Prérequis
- Python 3.8+
- pip
- Git

### Installation des dépendances
```bash
pip install -r requirements.txt
```

### Dépendances système (Linux/Ubuntu)
Certaines dépendances cartographiques nécessitent des packages système. Installez-les avec :
```bash
apt-get update && apt-get install -y gdal-bin python-gdal
```

## Utilisation
Pour lancer l'application :
```bash
streamlit run app.py
```

L'application sera accessible à l'adresse : http://localhost:8501

## Optimisations de performance

L'application a été optimisée pour améliorer les performances lors de la navigation entre les pages :

1. **Chargement paresseux des données** :
   - Les données sont chargées uniquement lorsqu'elles sont nécessaires
   - Utilisation de `st.session_state` pour stocker les données entre les rechargements

2. **Mise en cache des fonctions** :
   - Utilisation de `@st.cache_data` pour mettre en cache les résultats des fonctions coûteuses
   - Création de fonctions spécifiques pour chaque transformation de données

3. **Modularisation du code** :
   - Séparation des fonctions de chargement, de préparation et d'affichage des données
   - Utilisation de fonctions typées pour améliorer la lisibilité et la maintenabilité

4. **Optimisation des visualisations** :
   - Utilisation d'images préchargées pour les visualisations statiques
   - Mise en cache des graphiques générés dynamiquement

5. **Gestion efficace de l'état** :
   - Conservation de l'état de l'interface utilisateur entre les rechargements
   - Initialisation minimale au démarrage de l'application

## Structure du projet

```
Projet_Accidents/
├── app.py                              # Application Streamlit principale
├── requirements.txt                    # Dépendances Python
├── packages.txt                        # Dépendances système pour Streamlit Cloud
├── README.md                          # Documentation du projet
├── data/                              # Données du projet
│   ├── 20241031_accidents_Fr_19_22_AH_q.csv  # Données accidents 2019-2022
│   ├── X_train.csv                    # Features d'entraînement
│   ├── X_test.csv                     # Features de test
│   ├── y_train.csv                    # Labels d'entraînement
│   ├── y_test.csv                     # Labels de test
│   └── departements-france.csv        # Référentiel départements
├── models/                            # Modèles entraînés
│   ├── 20250129_catboost_best_model.pkl      # Modèle CatBoost
│   ├── 20250129_xgb_best_model.pkl           # Modèle XGBoost
│   ├── 20250304_XGBoost_bin_best_shap_values.pkl   # Valeurs SHAP XGBoost
│   └── 20250304_catboost_bin_best_shap_values.pkl  # Valeurs SHAP CatBoost
├── images/                            # Images et visualisations
│   └── ...                           # Graphiques et diagrammes
└── communes-20220101-shp/            # Données géographiques (shapefile)
    └── ...                           # Fichiers shapefile des communes

```

## Données utilisées

- **Accidents routiers** : Base de données des accidents corporels de la circulation routière (2019-2022)
- **Données géographiques** : Shapefile des communes françaises pour la cartographie
- **Données départementales** : Référentiel des départements français
- **Modèles ML** : Modèles pré-entraînés XGBoost et CatBoost avec leurs valeurs SHAP

## Auteur
Projet réalisé dans le cadre de la formation Data Science

## Licence
Ce projet est distribué sous licence MIT 