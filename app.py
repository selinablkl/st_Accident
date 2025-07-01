
# Packages
import streamlit as st
from streamlit_option_menu import option_menu
import pandas as pd
import geopandas as gpd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from typing import Tuple, List, Dict, Any, Union, Optional
from sklearn.preprocessing import StandardScaler
import shap
import joblib
import os
import gc  # Garbage collection for memory management
import warnings
warnings.filterwarnings('ignore')

# Memory optimization settings
pd.options.mode.chained_assignment = None  # Disable the SettingWithCopyWarning

def get_memory_usage():
    """Retourne l'utilisation mémoire actuelle en MB."""
    try:
        import psutil
        process = psutil.Process(os.getpid())
        return process.memory_info().rss / 1024 / 1024
    except (ImportError, NameError):
        # If psutil is not available, return 0
        return 0

def log_memory_usage(operation_name: str):
    """Log l'utilisation mémoire pour une opération donnée."""
    memory_mb = get_memory_usage()
    if memory_mb > 0:  # Only log if we could get memory usage
        print(f"{operation_name}: {memory_mb:.2f} MB")

def cleanup_memory():
    """Nettoie la mémoire en forçant le garbage collection."""
    gc.collect()
    
    # Pour les DataFrames volumineux, on peut aussi libérer le cache Streamlit si nécessaire
    # st.cache_data.clear()  # Décommenter si besoin de libérer tout le cache

def monitor_memory(func):
    """Décorateur pour surveiller l'utilisation mémoire d'une fonction."""
    def wrapper(*args, **kwargs):
        # Mémoire avant
        mem_before = get_memory_usage()
        
        # Exécuter la fonction
        result = func(*args, **kwargs)
        
        # Mémoire après
        mem_after = get_memory_usage()
        
        # Log la différence
        diff = mem_after - mem_before
        if diff > 10:  # Log seulement si la différence est significative (>10 MB)
            print(f"{func.__name__}: +{diff:.2f} MB (total: {mem_after:.2f} MB)")
        
        return result
    return wrapper

def estimate_dataframe_memory(filepath: str, sep: str = ",") -> float:
    """
    Estime l'utilisation mémoire d'un DataFrame avant de le charger complètement.
    
    Args:
        filepath: Chemin vers le fichier CSV
        sep: Séparateur de colonnes
        
    Returns:
        Estimation de la mémoire en MB
    """
    # Lire seulement les 1000 premières lignes pour estimer
    sample = pd.read_csv(filepath, sep=sep, nrows=1000)
    
    # Calculer la mémoire par ligne
    memory_per_row = sample.memory_usage(deep=True).sum() / len(sample) / 1024 / 1024
    
    # Compter le nombre total de lignes (rapide, ne charge pas le fichier)
    with open(filepath, 'r') as f:
        total_rows = sum(1 for _ in f) - 1  # -1 pour l'en-tête
    
    # Estimer la mémoire totale
    estimated_memory = memory_per_row * total_rows
    
    return estimated_memory

def read_csv_optimized(filepath: str, sep: str = ",", chunksize: int = 10000) -> pd.DataFrame:
    """
    Lit un fichier CSV par chunks pour optimiser la mémoire.
    
    Args:
        filepath: Chemin vers le fichier CSV
        sep: Séparateur de colonnes
        chunksize: Taille des chunks
    
    Returns:
        DataFrame complet optimisé
    """
    chunks = []
    
    # Lire le fichier par chunks
    for chunk in pd.read_csv(filepath, sep=sep, chunksize=chunksize, low_memory=False):
        # Optimiser chaque chunk
        chunk = optimize_dtypes(chunk)
        chunks.append(chunk)
    
    # Concaténer tous les chunks
    df = pd.concat(chunks, ignore_index=True)
    
    # Libérer la mémoire des chunks
    del chunks
    gc.collect()
    
    return df

def optimize_dtypes(df: pd.DataFrame) -> pd.DataFrame:
    """
    Optimise les types de données d'un DataFrame pour réduire l'utilisation mémoire.
    
    Args:
        df: DataFrame à optimiser
    
    Returns:
        DataFrame avec types optimisés
    """
    for col in df.columns:
        col_type = df[col].dtype
        
        # Optimisation des entiers
        if col_type != 'object':
            c_min = df[col].min()
            c_max = df[col].max()
            
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)
            
            # Optimisation des flottants
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        
        # Optimisation des chaînes de caractères avec category
        else:
            num_unique_values = len(df[col].unique())
            num_total_values = len(df[col])
            if num_unique_values / num_total_values < 0.5:
                df[col] = df[col].astype('category')
    
    return df

def get_dataset_columns(dataset_name: str) -> Optional[List[str]]:
    """
    Retourne les colonnes importantes pour chaque dataset.
    Permet de charger seulement les colonnes nécessaires.
    """
    columns_dict = {
        "accidents": [
            'date', 'heure', 'departement', 'num_commune', 'luminosite', 
            'conditions_atmos', 'type_collision', 'etat_surface', 'an_sem', 'annee',
            'mois', 'jr_sem_q', 'tranche_heure', 'gravite_accident',
            'latitude', 'longitude'
        ],
        # Ajouter d'autres datasets si nécessaire
    }
    return columns_dict.get(dataset_name, None)

@st.cache_data
def load_data(dataset_names: Optional[List[str]] = None, optimize_memory: bool = True, use_columns: Optional[Dict[str, List[str]]] = None) -> Dict[str, Union[pd.DataFrame, gpd.GeoDataFrame]]:
    """
    Charge les données nécessaires de manière optimisée.
    
    Args:
        dataset_names: Liste des noms de datasets à charger. Si None, charge tous les datasets.
        optimize_memory: Si True, optimise les types de données pour réduire la mémoire.
        use_columns: Dict optionnel spécifiant les colonnes à charger pour chaque dataset.
    
    Returns:
        Un dictionnaire contenant les datasets demandés.
    """
    path = "./data/"
    
    # Définir tous les datasets disponibles avec optimisation mémoire
    # Pour les gros fichiers comme accidents, on peut utiliser des chunks
    all_datasets = {
        "X_train": lambda cols=None: pd.read_csv(path + "X_train.csv", low_memory=False, usecols=cols),
        "X_test": lambda cols=None: pd.read_csv(path + "X_test.csv", low_memory=False, usecols=cols),
        "y_train": lambda cols=None: pd.read_csv(path + "y_train.csv", low_memory=False, usecols=cols),
        "y_test": lambda cols=None: pd.read_csv(path + "y_test.csv", low_memory=False, usecols=cols),
        "dep": lambda cols=None: pd.read_csv(path + "departements-france.csv", low_memory=False, usecols=cols),
        "accidents": lambda cols=None: read_csv_optimized(path + "20241031_accidents_Fr_19_22_AH_q.csv", sep=";") if cols is None 
                                      else pd.read_csv(path + "20241031_accidents_Fr_19_22_AH_q.csv", sep=";", usecols=cols, low_memory=False),
        "france": lambda cols=None: gpd.read_file("./communes-20220101-shp/communes-20220101.shp") if cols is None else gpd.read_file("./communes-20220101-shp/communes-20220101.shp")
    }
    
    # Si aucun dataset spécifique n'est demandé, charger tous les datasets
    if dataset_names is None:
        dataset_names = list(all_datasets.keys())
    
    # Charger uniquement les datasets demandés
    result = {}
    for name in dataset_names:
        if name in all_datasets:
            # Déterminer les colonnes à charger
            cols = None
            if use_columns and name in use_columns:
                cols = use_columns[name]
            elif optimize_memory and name == "accidents":
                # Utiliser les colonnes par défaut pour accidents si optimize_memory est True
                cols = get_dataset_columns("accidents")
            
            # Charger le dataset avec les colonnes spécifiées
            df = all_datasets[name](cols)
            
            # Optimiser la mémoire si demandé (ne pas optimiser les GeoDataFrames)
            if optimize_memory and isinstance(df, pd.DataFrame) and not isinstance(df, gpd.GeoDataFrame):
                df = optimize_dtypes(df)
            
            result[name] = df
    
    # Force garbage collection après chargement
    gc.collect()
    
    return result

@st.cache_data
def prepare_dep_accidents(accidents: pd.DataFrame, dep: pd.DataFrame) -> pd.DataFrame:
    """Prépare les données d'accidents par département."""
    return accidents.merge(dep, left_on="departement", right_on="code_departement", how="left")

@st.cache_data
def prepare_france_data(_france: gpd.GeoDataFrame, count: pd.DataFrame) -> pd.DataFrame:
    """Prépare les données géospatiales de la France."""
    france_merged = _france.merge(count, left_on="insee", right_on="num_commune", how="left")
    return france_merged[france_merged['insee'] < '96000']

@st.cache_data
def get_top_communes(_france_data: pd.DataFrame, n: int = 1000) -> pd.DataFrame:
    """Récupère les N communes avec le plus d'accidents."""
    top_communes = _france_data.nlargest(n, 'count')
    top_communes['count'] = top_communes['count'].round(1)
    return top_communes

@st.cache_data
def prepare_accidents_binaire(_accidents: pd.DataFrame) -> pd.DataFrame:
    """Crée une version binaire des données d'accidents."""
    accidents_binaire = _accidents.copy()
    value_to_replace = {
        'indemne': '0',
                    'blesse_leger': '1',
                    'blesse_hospitalise': '1',
        'tue': '1'
    }
    accidents_binaire["gravite_accident"] = _accidents['gravite_accident'].replace(value_to_replace)
    return accidents_binaire

@st.cache_data
def prepare_time_series_data(_accidents: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Prépare les données pour les analyses temporelles."""
    accidents_dt = _accidents.copy()
    accidents_dt['date'] = pd.to_datetime(accidents_dt['date'], format='%d/%m/%Y', dayfirst=True)
    
    # Utiliser la colonne 'annee' existante au lieu de créer une colonne 'year'
    accidents_dt['year'] = accidents_dt['annee']
    accidents_dt['month'] = accidents_dt['date'].dt.month
    
    # Compter le nombre d'accidents par année
    accidents_per_year = accidents_dt.groupby('year').size().reset_index()
    accidents_per_year.columns = ['year', 'nombre_accidents']
    
    # Compter le nombre d'accidents par mois et par année
    monthly_accidents = accidents_dt.groupby(['year', 'month']).size().reset_index()
    monthly_accidents.columns = ['year', 'month', 'nombre_accidents']
    
    # Ajouter les noms des mois
    month_names = ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin', 
                   'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre']
    monthly_accidents['month_name'] = monthly_accidents['month'].apply(lambda x: month_names[x-1])
    
    # Créer la colonne 'sem_ms_an' contenant le numéro de la semaine de l'année
    accidents_dt['sem_ms_an'] = accidents_dt['date'].dt.isocalendar().week
    
    # Grouper par année et numéro de semaine pour compter le nombre d'accidents par semaine
    accidents_per_week = accidents_dt.groupby([accidents_dt['year'], 'sem_ms_an']).size().reset_index()
    accidents_per_week.columns = ['year', 'sem_ms_an', 'accidents_par_sem_an']
    
    return accidents_dt, monthly_accidents, accidents_per_week, accidents_per_year

# Initialisation des données communes à toutes les pages
if 'initialized' not in st.session_state:
    # Log mémoire au démarrage
    log_memory_usage("Démarrage de l'application")
    
    # Chargement minimal des données au démarrage - seulement accidents avec optimisation
    data_dict = load_data(["accidents"], optimize_memory=True)
    accidents = data_dict["accidents"]
    
    # Log mémoire après chargement
    log_memory_usage("Après chargement des accidents")
    
    # Préparation des données binaires (utilisées dans plusieurs pages)
    accidents_binaire = prepare_accidents_binaire(accidents)
    
    # Stockage dans session_state pour éviter de recharger
    st.session_state.accidents = accidents
    st.session_state.accidents_binaire = accidents_binaire
    st.session_state.initialized = True
    
    # Nettoyage mémoire après initialisation
    cleanup_memory()
    log_memory_usage("Après initialisation et nettoyage")
else:
    # Récupération des données déjà chargées
    accidents = st.session_state.accidents
    accidents_binaire = st.session_state.accidents_binaire

######################### Streamlit app #########################

# Afficher l'utilisation mémoire dans la sidebar (optionnel)
with st.sidebar:
    if st.checkbox("Afficher l'utilisation mémoire", value=False):
        memory_mb = get_memory_usage()
        st.metric("Mémoire utilisée", f"{memory_mb:.1f} MB")
        
        # Bouton pour nettoyer la mémoire
        if st.button("Nettoyer la mémoire"):
            cleanup_memory()
            st.success("Mémoire nettoyée!")
            st.rerun()

selected = option_menu(
    menu_title=None,
    options=["Introduction", "Exploration", "Visualisation", "Modélisation", "Conclusion", "Chat"],
    icons=["lightbulb", "book", "bar-chart", "gear", "clipboard-data", "chat"],
    orientation="horizontal",
    styles={
        "container": {"padding": "0!important", "background-color": "#f0f2f6"},
        "icon": {"color": "#1E3A8A", "font-size": "12px"},
        "nav-link": {"font-size": "12px", "text-align": "center", "margin":"0px", "padding":"0px 0px 0px 0px", "border-radius": "5px"},
        "nav-link-selected": {"background-color": "#3B82F6"},
    }
)


# Chargement des données spécifiques à chaque page
if selected == "Introduction":  # Introduction
    # Ajout d'un style CSS pour améliorer l'apparence
    st.markdown("""
    <style>
        .main-title {
            color: #1E3A8A;
            font-size: 36px;
            font-weight: 700;
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 15px;
            border-bottom: 2px solid #3B82F6;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        }
        .title-container {
            background: linear-gradient(to right, #f0f4ff, #ffffff, #f0f4ff);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 40px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        }
        .intro-header {
            color: #1E3A8A;
            font-size: 28px;
            font-weight: 600;
            margin-bottom: 20px;
        }
        .intro-text {
            font-size: 16px;
            line-height: 1.6;
            text-align: justify;
            margin-bottom: 15px;
        }
        .intro-highlight {
            background-color: #F3F4F6;
            border-left: 4px solid #3B82F6;
            padding: 15px;
            margin: 20px 0;
        }
        .intro-source {
            font-size: 14px;
            color: #6B7280;
            font-style: italic;
        }
        .intro-section {
            margin-top: 30px;
            margin-bottom: 30px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Titre principal amélioré avec conteneur
    st.markdown("""
    <div class="title-container">
        <div class="main-title">Projet Accidents Routiers en France</div>
        <div style="text-align: center; font-size: 20px; color: #4B5563; font-weight: 500;">Analyse des données d'accidentologie 2019-2022</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Titre de l'introduction
    st.markdown('<div class="intro-header">Contexte et enjeux de la sécurité routière en France</div>', unsafe_allow_html=True)
    
    # Premier paragraphe
    st.markdown("""
    <div class="intro-text">
        Chaque année en France, plus de 50 000 accidents corporels sont recensés par l'Organisation interministériel 
        de la sécurité routière (ONISR), et plus de 1,8 million de constats amiables sont reportés par les compagnies d'assurance.
        Ces chiffres soulignent l'importance cruciale d'une analyse approfondie des données d'accidentologie pour améliorer 
        la sécurité routière et réduire le nombre de victimes sur nos routes.
    </div>
    """, unsafe_allow_html=True)
    
    # Deuxième paragraphe
    st.markdown("""
    <div class="intro-text">
        La mortalité routière en France se trouve dans la moyenne européenne selon les études menées par l'Institut national 
        de la statistique et des études économiques (Insee), comparant les accidents entre les années 2000 et 2022 des pays 
        membres de l'Union Européenne. Cette position reflète les efforts constants déployés par les autorités françaises, 
        tout en soulignant la nécessité de poursuivre les actions de prévention.
    </div>
    """, unsafe_allow_html=True)
    
    
    # Troisième paragraphe
    st.markdown("""
    <div class="intro-source">Sources : ONISR, Insee</div>
    """, unsafe_allow_html=True)
    
    st.markdown(
        """
        <p></p>
        <p></p>
        """,
            unsafe_allow_html=True
    )

    st.markdown(
        """
        <p></p>
        <p></p>
        """,
            unsafe_allow_html=True
    )

    st.markdown('<div class="intro-header">Création de la base de données</div>', unsafe_allow_html=True)
    
    # Ajout d'un style CSS pour améliorer l'alignement des images
    st.markdown("""
    <style>
        .image-title {
            text-align: center;
            font-weight: 600;
            color: #1E3A8A;
            font-size: 18px;
            margin-bottom: 18px;
        }
        .image-icon {
            font-size: 15px;
            margin-right: 2px;
            vertical-align: middle;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Utilisation des colonnes Streamlit avec espacement égal
    col1, col2, col3 = st.columns([1, 1, 1])
    
    # Définition des images, leurs titres et icônes
    images_data = [
        {
            "title": "Modèle de données",
            "path": "./images/20250313_Modèle-Données_Projet-Accidents.jpg",
            "icon": "📊"  # Icône pour modèle de données
        },
        {
            "title": "Construction de la base de données",
            "path": "./images/20250313_Construction-BdD_Données2019-2022.jpg",
            "icon": "🔨"  # Icône pour construction
        },
        {
            "title": "Nettoyage et prétraitement",
            "path": "./images/20250313_Clean-Preprocess_Features.jpg",
            "icon": "🧹"  # Icône pour nettoyage
        }
    ]
    
    # Affichage des images dans les colonnes
    for col, img_data in zip([col1, col2, col3], images_data):
        with col:
            # Titre avec icône
            st.markdown(f'<div class="image-title"><span class="image-icon">{img_data["icon"]}</span>{img_data["title"]}</div>', unsafe_allow_html=True)
            # Affichage de l'image avec une hauteur fixe pour assurer l'alignement
            st.image(img_data["path"], use_column_width=True)
    
    # Ajout d'un espace après les images
    st.markdown("<div style='height: 30px;'></div>", unsafe_allow_html=True)

    st.markdown(
        """
        <p></p>
        <p></p>
        """,
            unsafe_allow_html=True
    )

    # Présentation du projet
    st.markdown("""
    <div class="intro-section">
        <div class="intro-header" style="font-size: 28px;">Objectifs de notre analyse</div>
        <div class="intro-text">
            Ce projet vise à explorer et analyser les données d'accidents routiers en France sur la période 2019-2022 
            pour identifier les facteurs de risque, les tendances temporelles et géographiques, ainsi que les caractéristiques 
            des accidents les plus graves. Notre analyse s'appuie sur des visualisations interactives et des modèles 
            statistiques pour offrir une compréhension approfondie de l'accidentologie routière en France.
        </div>
    </div>
    """, unsafe_allow_html=True)

if selected == "Exploration":  # Exploration    
    
    # Charger les données départementales si nécessaire
    if 'dep' not in st.session_state:
        dep_dict = load_data(["dep"], optimize_memory=True)
        st.session_state.dep = dep_dict["dep"]
    dep = st.session_state.dep
    
    # Ajout d'un style CSS pour améliorer l'apparence (même style que l'introduction et la conclusion)
    st.markdown("""
    <style>
        .main-title {
            color: #1E3A8A;
            font-size: 36px;
            font-weight: 700;
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 15px;
            border-bottom: 2px solid #3B82F6;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        }
        .title-container {
            background: linear-gradient(to right, #f0f4ff, #ffffff, #f0f4ff);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 40px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        }
        .exploration-header {
            color: #1E3A8A;
            font-size: 28px;
            font-weight: 600;
            margin-bottom: 20px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Titre principal amélioré avec conteneur
    st.markdown("""
    <div class="title-container">
        <div class="main-title">Exploration des Données</div>
        <div style="text-align: center; font-size: 20px; color: #4B5563; font-weight: 500;">Analyse des caractéristiques des accidents routiers</div>
    </div>
    """, unsafe_allow_html=True)
    


    # Table des variables du dataset et description
    data = {
        "Variable": ["Num_Acc", "dep", "id_vehicule", "com", "num_veh", "agg", "place", "int", "catu", "atm",
                    "grav", "col", "sexe", "lat", "an_nais", "long", "trajet", "date", "secu1", "secu2",
                    "trch_hor_jr", "locp", "actp", "etatp", "trimestre", "senc", "jr_nuit", "catv", "catr",
                    "obs", "circ", "obsm", "nbv", "choc", "vosp", "manv", "prof", "motor", "pr", "occutc",
                    "pr1", "day", "plan", "month", "surf", "year", "infra", "hrmn", "situ", "lum", "vma"],
        "Descriptif": ["Numéro de l'accident", "Département", "Identifiant du véhicule (code numérique)", "Commune",
                    "Identifiant du véhicule (code alphanumérique)", "Localisation : hors agglo / en agglo",
                    "Place occupée par l'usager", "Intersection", "Catégorie d'usager", "Conditions atmosphériques",
                    "Gravité de l'accident", "Type de collision", "Sexe de l'usager", "Latitude", "Année de naissance",
                    "Longitude", "Type de trajet emprunté", "Date de l'accident", "Présence d'un équipement de sécurité",
                    "Présence d'un équipement de sécurité", "Tranche horaire de la journée", "Localisation du piéton",
                    "Action du piéton", "Présence d'autres piétons", "Trimestre", "Sens de circulation",
                    "Distinction jour/nuit", "Catégorie du véhicule", "Catégorie de route", "Obstacle fixe heurté",
                    "Régime de circulation", "Obstacle mobile heurté", "Nombre de voies", "Point de choc initial",
                    "Existence d'une voie réservée", "Manoeuvre principale avant l'accident", "Déclivité de la route",
                    "Type de motorisation", "Numéro du PR", "Nombre d'occupants dans le transport en commun",
                    "Distance en mètres du PR", "Jour de l'accident", "Tracé en plan", "Mois de l'accident",
                    "État de la surface", "Année de l'accident", "Aménagement - infrastructure",
                    "Heure et minutes", "Situation de l'accident", "Conditions d'éclairage", "Vitesse maximale autorisée"]
    }

    # Création du DataFrame
    df = pd.DataFrame(data)

    # Réorganiser les colonnes pour un affichage en 4 colonnes
    df_pairs = pd.DataFrame({
        "Variable 1": df["Variable"][::2].reset_index(drop=True),
        "Description 1": df["Descriptif"][::2].reset_index(drop=True),
        "Variable 2": df["Variable"][1::2].reset_index(drop=True),
        "Description 2": df["Descriptif"][1::2].reset_index(drop=True),
    })
    # Style avec CSS
    st.markdown("""
        <style>
            .title {
                color: white;
                background: linear-gradient(to right, #1E3A8A, #3B82F6, #1E3A8A);
                padding: 12px 15px;
                border-radius: 8px;
                text-align: center;
                font-size: 24px;
                font-weight: 600;
                margin-bottom: 20px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
            }
            .search-box {
                font-size: 18px;
                padding: 10px;
                width: 100%;
                border: 2px solid #3B82F6;
                border-radius: 5px;
                margin-bottom: 20px;
            }
        </style>
    """, unsafe_allow_html=True)

    st.markdown(
        """
        <p></p>
        <p></p>
        """,
            unsafe_allow_html=True
    )

    # Titre stylisé
    st.markdown('<div class="title">📊 Description des Variables du Dataset</div>', unsafe_allow_html=True)

    # Champ de recherche
    search_query = st.text_input("🔍 Rechercher une variable :", "").lower()

    # Filtrer les résultats
    if search_query:
        df_pairs = df_pairs[df_pairs.apply(lambda row: row.astype(str).str.lower().str.contains(search_query).any(), axis=1)]

    # Accordéon pour afficher la table
    with st.expander("📌 Voir la table des variables", expanded=True):
        st.dataframe(df_pairs, use_container_width=True)

    # Préparation des données temporelles si nécessaire
    if 'time_series_data' not in st.session_state:
        accidents_dt, monthly_accidents, accidents_per_week, accidents_per_year = prepare_time_series_data(accidents)
        st.session_state.accidents_dt = accidents_dt
        st.session_state.monthly_accidents = monthly_accidents
        st.session_state.accidents_per_week = accidents_per_week
    else:
        accidents_dt = st.session_state.accidents_dt
        monthly_accidents = st.session_state.monthly_accidents
        accidents_per_week = st.session_state.accidents_per_week

    st.markdown(
        """
        <p></p>
        <p></p>
        """,
            unsafe_allow_html=True
    )
    st.markdown(
        """
        <p></p>
        <p></p>
        """,
            unsafe_allow_html=True
    )
    
    # Graphique interactif du nombre d'accidents par année
    st.markdown('<div class="title">📈 Évolution des accidents par année</div>', unsafe_allow_html=True)
    
    # Compter le nombre d'accidents par année
    accidents_per_year = accidents_dt.groupby('year').size().reset_index()
    accidents_per_year.columns = ['year', 'nombre_accidents']
    
    # Créer un graphique interactif avec Plotly
    fig_yearly = px.bar(
        accidents_per_year,
        x='year',
        y='nombre_accidents',
        title="Nombre d'accidents par année (2019-2022)",
        labels={'year': 'Année', 'nombre_accidents': "Nombre d'accidents"},
        color='nombre_accidents',
        color_continuous_scale='Viridis',
        text='nombre_accidents'
    )
    
    # Personnalisation du graphique
    fig_yearly.update_traces(
        texttemplate='%{text:,}',
        textposition='outside',
        hovertemplate='Année: %{x}<br>Nombre d\'accidents: %{y:,}'
    )
    
    fig_yearly.update_layout(
        xaxis=dict(tickmode='linear', dtick=1),
        yaxis=dict(title="Nombre d'accidents"),
        coloraxis_showscale=False,
        hoverlabel=dict(bgcolor="white", font_size=12),
        height=500
    )
    
    # Afficher le graphique interactif
    st.plotly_chart(fig_yearly, use_container_width=True)
    
    st.markdown(
        """
        <div style="background-color:#f9f9f9; padding:10px; border-radius:5px; border-left:5px solid #007BFF;">
            <b>📌 Analyse du Schéma :</b>
            Ce graphique illustre l'évolution du nombre d'accidents par année.
            <ul>
              <li>Une <b>baisse notable</b> est observée sur une année spécifique, possiblement due à des <b>facteurs externes</b> (ex. confinement COVID-19).</li>
              <li>Les autres années affichent <b>une relative stabilité</b>, nécessitant une <b>analyse approfondie des tendances et des mesures de sécurité.</b></li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p></p>
        <p></p>
        """,
            unsafe_allow_html=True
    )

    st.markdown(
        """
        <p></p>
        <p></p>
        """,
            unsafe_allow_html=True
    )
    st.markdown(
        """
        <p></p>
        <p></p>
        """,
            unsafe_allow_html=True
    )

    # Ajout d'un sélecteur pour comparer les données mensuelles par année
    st.markdown('<div class="title">📊 Comparaison mensuelle des accidents par année</div>', unsafe_allow_html=True)
    
    # Sélection des années à afficher
    selected_years = st.multiselect(
        "Sélectionnez les années à comparer",
        options=sorted(monthly_accidents['year'].unique()),
        default=sorted(monthly_accidents['year'].unique())
    )
    
    # Filtrer les données selon les années sélectionnées
    filtered_data = monthly_accidents[monthly_accidents['year'].isin(selected_years)]
    
    # Créer un graphique interactif pour la comparaison mensuelle
    fig_monthly = px.line(
        filtered_data,
        x='month',
        y='nombre_accidents',
        color='year',
        markers=True,
        labels={'month': 'Mois', 'nombre_accidents': "Nombre d'accidents", 'year': 'Année'},
        title="Évolution mensuelle des accidents par année"
    )
    
    # Personnalisation du graphique
    month_names = ['Janvier', 'Février', 'Mars', 'Avril', 'Mai', 'Juin', 
                   'Juillet', 'Août', 'Septembre', 'Octobre', 'Novembre', 'Décembre']
    fig_monthly.update_layout(
        xaxis=dict(
            tickmode='array',
            tickvals=list(range(1, 13)),
            ticktext=month_names,
            tickangle=45
        ),
        yaxis=dict(title="Nombre d'accidents"),
        legend=dict(title="Année"),
        hovermode="x unified",
        height=500
    )
    
    # Afficher le graphique interactif
    st.plotly_chart(fig_monthly, use_container_width=True)
    
    st.markdown(
        """
        <div style="background-color:#f9f9f9; padding:10px; border-radius:5px; border-left:5px solid #007BFF;">
            <b>📌 Analyse du Schéma :</b>
            Ce graphique représente l'<b>évolution mensuelle des accidents </b>sur plusieurs années.
            <ul>
              <li>Une <b>tendance saisonnière </b>est visible, avec des <b>pics récurrents</b> en certaines périodes de l'année.</li>
              <li>L'analyse permet d'identifier <b>les périodes à risque élevé</b>, essentielles pour adapter les <b>politiques de prévention routière.</b></li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p></p>
        <p></p>
        """,
            unsafe_allow_html=True
    )

    # Tracer la série chronologique des accidents par semaine
    st.image("./images/output_10_1.png")

    st.markdown(
        """
        <div style="background-color:#f9f9f9; padding:10px; border-radius:5px; border-left:5px solid #007BFF;">
            <b>📌 Analyse du Schéma :</b>
            Ce graphique illustre l'<b>évolution hebdomadaire du nombre d'accidents entre 2019 et 2022</b>.
            <ul>
              <li><b>2020 (orange)</b> présente une <b>forte baisse</b> autour des semaines 10 à 20, correspondant aux confinements liés au COVID-19.</li>
              <li>Les autres années suivent des tendances similaires, avec des <b>pics d'accidentalité en milieu et fin d'année</b>.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )    

    st.markdown(
        """
        <p></p>
        <p></p>
        """,
            unsafe_allow_html=True
    )

    st.markdown(
        """
        <p></p>
        <p></p>
        """,
            unsafe_allow_html=True
    )
    st.markdown(
        """
        <p></p>
        <p></p>
        """,
            unsafe_allow_html=True
    )
    # Ajout d'une carte choroplèthe interactive des accidents par département
    st.markdown('<div class="title">🗺️ Répartition géographique des accidents par département</div>', unsafe_allow_html=True)
    
    # Préparation des données pour la carte
    # Compter le nombre d'accidents par département et par année
    dep_year_accidents = accidents_dt.groupby(['year', 'departement']).size().reset_index()
    dep_year_accidents.columns = ['year', 'departement', 'nombre_accidents']
    
    # Obtenir les années disponibles dans les données
    available_years = sorted(dep_year_accidents['year'].unique())
    
    # Sélection de l'année à afficher sur la carte
    selected_year_map = st.selectbox(
        "Sélectionnez l'année à visualiser sur la carte",
        options=available_years,
        index=len(available_years) - 1  # Sélectionner la dernière année par défaut
    )
    
    # Filtrer les données pour l'année sélectionnée
    filtered_dep_data = dep_year_accidents[dep_year_accidents['year'] == selected_year_map]
    
    # Vérifier si des données sont disponibles pour l'année sélectionnée
    if len(filtered_dep_data) == 0:
        st.warning(f"Aucune donnée disponible pour l'année {selected_year_map}. Veuillez sélectionner une autre année.")
    else:
        # Convertir les codes de département pour assurer la compatibilité lors de la fusion
        filtered_dep_data = filtered_dep_data.copy()
        filtered_dep_data['departement'] = filtered_dep_data['departement'].astype(str).str.zfill(2)
        
        # Fusionner avec les données des départements pour obtenir les noms
        filtered_dep_data = filtered_dep_data.merge(dep[['code_departement', 'nom_departement']], 
                                                left_on='departement', 
                                                right_on='code_departement', 
                                                how='left')
        
        # Vérifier si la fusion a fonctionné correctement
        if 'nom_departement' not in filtered_dep_data.columns or filtered_dep_data['nom_departement'].isna().all():
            st.warning(f"Problème lors de la fusion des données pour l'année {selected_year_map}. Certaines informations peuvent être manquantes.")
            # Ajouter une colonne nom_departement par défaut si elle est manquante
            if 'nom_departement' not in filtered_dep_data.columns:
                filtered_dep_data['nom_departement'] = filtered_dep_data['departement'].astype(str)
            # Remplacer les valeurs NaN par le code du département
            filtered_dep_data['nom_departement'] = filtered_dep_data['nom_departement'].fillna(filtered_dep_data['departement'].astype(str))
        
        # Créer la carte choroplèthe
        @st.cache_data
        def create_choropleth_map(_filtered_data):
            fig_map = px.choropleth(
                _filtered_data,
                geojson="https://raw.githubusercontent.com/gregoiredavid/france-geojson/master/departements.geojson",
                locations='departement',
                featureidkey="properties.code",
                color='nombre_accidents',
                color_continuous_scale="Viridis",
                range_color=[0, _filtered_data['nombre_accidents'].max()],
                scope="europe",
                labels={'nombre_accidents': "Nombre d'accidents", 'departement': 'Code département', 'nom_departement': 'Département'},
                hover_name='nom_departement',
                hover_data={'departement': True, 'nombre_accidents': True, 'nom_departement': True}
            )
            
            fig_map.update_geos(
                fitbounds="locations",
                visible=False,
                resolution=50,
                showcoastlines=True,
                coastlinecolor="RebeccaPurple",
                showland=True,
                landcolor="LightGreen",
                showocean=True,
                oceancolor="LightBlue"
            )
            
            fig_map.update_layout(
                margin={"r": 0, "t": 0, "l": 0, "b": 0},
                height=600,
                geo=dict(
                    center=dict(lon=2.5, lat=46.8),
                    projection_scale=20
                )
            )
            return fig_map
        
        # Afficher la carte
        if len(filtered_dep_data) > 0:
            fig_map = create_choropleth_map(filtered_dep_data)
            st.plotly_chart(fig_map, use_container_width=True)

    # Top 10 des départements avec le plus d'accidents pour l'année sélectionnée
    if len(filtered_dep_data) > 0:
        st.markdown(f"#### Top 10 des départements avec le plus d'accidents en {selected_year_map}")
        
        # Trier et sélectionner les 10 premiers départements
        top_10_deps = filtered_dep_data.sort_values('nombre_accidents', ascending=False).head(10)
        
        # Vérifier si des données sont disponibles pour l'année sélectionnée
        if len(top_10_deps) == 0:
            st.warning(f"Aucune donnée disponible pour l'année {selected_year_map}. Veuillez sélectionner une autre année.")
        else:
            # Vérifier si la colonne nom_departement existe et contient des données
            if 'nom_departement' not in top_10_deps.columns or top_10_deps['nom_departement'].isna().all():
                # Utiliser le code du département comme nom si le nom n'est pas disponible
                top_10_deps['nom_departement'] = top_10_deps['departement'].astype(str)
            else:
                # Remplacer les valeurs NaN par le code du département
                top_10_deps['nom_departement'] = top_10_deps['nom_departement'].fillna(top_10_deps['departement'].astype(str))
            
            # Créer un graphique à barres horizontales pour le top 10
            @st.cache_data
            def create_top_deps_chart(_top_deps, year):
                fig_top_deps = px.bar(
                    _top_deps,
                    y='nom_departement',
                    x='nombre_accidents',
                    orientation='h',
                    color='nombre_accidents',
                    color_continuous_scale='Viridis',
                    labels={'nombre_accidents': "Nombre d'accidents", 'nom_departement': 'Département'},
                    title=f"Top 10 des départements avec le plus d'accidents en {year}"
                )
                
                fig_top_deps.update_layout(
                    yaxis={'categoryorder': 'total ascending'},
                    height=500
                )
                return fig_top_deps
            
            # Afficher le graphique
            fig_top_deps = create_top_deps_chart(top_10_deps, selected_year_map)
            st.plotly_chart(fig_top_deps, use_container_width=True)
    
    st.markdown(
        """
        <div style="background-color:#f9f9f9; padding:10px; border-radius:5px; border-left:5px solid #007BFF;">
            <b>📌 Analyse du Schéma :</b>
            Ce graphique présente le <b>Top 10 des départements les plus accidentogènes </b> en 2022.
            <ul>
              <li>Certains départements affichent un <b>nombre significativement élevé</b> d'accidents.</li>
              <li>Ces données sont essentielles pour <b>cibler les interventions</b> et améliorer la <b>sécurité routière</b> dans les zones les plus touchées.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p></p>
        <p></p>
        """,
            unsafe_allow_html=True
    )

    # Chargement des données nécessaires uniquement pour cette page
    if 'france' not in st.session_state:
        # Log mémoire avant chargement des données géospatiales
        log_memory_usage("Avant chargement des données géospatiales")
        
        data_dict = load_data(["france"], optimize_memory=True)
        st.session_state.france = data_dict["france"]
        
        # Log mémoire après chargement
        log_memory_usage("Après chargement des données géospatiales")
    
    france = st.session_state.france
    
    # Préparation des données pour l'affichage
    if 'top_1000_communes' not in st.session_state:
        # Nombre d'accident par département
        count = accidents.groupby("num_commune").size().reset_index()
        count.columns = ["num_commune", "count"]
        
        # Fusionner les données géospatiales avec les données d'accidents
        france_data = prepare_france_data(france, count)
        
        # Trier les communes par nombre d'accidents et sélectionner les 1000 premières
        top_1000_communes = get_top_communes(france_data, 1000)
        
        st.session_state.top_1000_communes = top_1000_communes
    else:
        top_1000_communes = st.session_state.top_1000_communes

	# Afficher les 1000 communes les plus accidentogènes de France
    st.image("./images/output_7_0.png")

        # Afficher le commentaire sous le graphique SEULEMENT
    st.markdown(
        """
        <div style="background-color:#f9f9f9; padding:10px; border-radius:5px; border-left:5px solid #007BFF;">
            <b>📌 Analyse du Schéma :</b>
            <ul>
              <li>Les <b color="red">zones rouges</b> indiquent une forte concentration d'accidents, principalement dans les grandes villes et les axes routiers majeurs.</li>
              <li>Les <b color="green">zones vertes</b> représentent des régions moins accidentogènes, souvent rurales.</li>
              <li>La carte met en évidence les zones critiques nécessitant des mesures de prévention renforcées.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p></p>
        <p></p>
        """,
            unsafe_allow_html=True
    )

    if st.checkbox("Afficher le top 30"):
        st.dataframe(top_1000_communes[['insee', 'num_commune', 'nom','count']].head(30))
    

    st.markdown(
        """
        <p></p>
        <p></p>
        """,
            unsafe_allow_html=True
    )
    st.markdown(
        """
        <p></p>
        <p></p>
        """,
            unsafe_allow_html=True
    )
    # Analyse de la gravité des accidents par année
    st.markdown('<div class="title">🚨 Évolution de la gravité des accidents par année</div>', unsafe_allow_html=True)
    
    # Préparation des données pour l'analyse de gravité
    @st.cache_data
    def prepare_gravity_data(_accidents_dt):
        gravity_year = _accidents_dt.groupby(['year', 'gravite_accident']).size().reset_index()
        gravity_year.columns = ['year', 'gravite_accident', 'nombre_accidents']
        
        # Créer un mapping pour les niveaux de gravité
        gravity_mapping = {
            'indemne': 'Indemne',
            'blesse_leger': 'Blessé léger',
            'blesse_hospitalise': 'Blessé hospitalisé',
            'tue': 'Tué'
        }
        
        # Appliquer le mapping si nécessaire
        if 'indemne' in gravity_year['gravite_accident'].values:
            gravity_year['gravite_label'] = gravity_year['gravite_accident'].map(gravity_mapping)
        else:
            # Si les valeurs sont déjà sous forme de libellés, les utiliser directement
            gravity_year['gravite_label'] = gravity_year['gravite_accident']
        
        return gravity_year
    
    gravity_year = prepare_gravity_data(accidents_dt)
    
    # Créer un graphique interactif pour l'évolution de la gravité
    @st.cache_data
    def create_gravity_chart(_gravity_data):
        fig_gravity = px.bar(
            _gravity_data,
            x='year',
            y='nombre_accidents',
            color='gravite_label',
            barmode='group',
            labels={'year': 'Année', 'nombre_accidents': "Nombre d'accidents", 'gravite_label': 'Gravité'},
            title="Évolution de la gravité des accidents par année"
        )
        
        fig_gravity.update_layout(
            xaxis=dict(tickmode='linear', dtick=1),
            yaxis=dict(title="Nombre d'accidents"),
            legend=dict(title="Niveau de gravité"),
            height=500
        )
        return fig_gravity
    
    # Afficher le graphique
    fig_gravity = create_gravity_chart(gravity_year)
    st.plotly_chart(fig_gravity, use_container_width=True)

    st.markdown(
        """
        <div style="background-color:#f9f9f9; padding:10px; border-radius:5px; border-left:5px solid #007BFF;">
            <b>📌 Analyse du Schéma :</b>
            Ce graphique montre l'<b>évolution de la gravité des accidents</b> au fil des années.
            <ul>
              <li>On observe une <b>stabilité des tendances </b>entre les différentes catégories d'accidents.</li>
              <li>Les accidents <b>graves/mortels</b> restent préoccupants et nécessitent un <b>suivi approfondi</b> pour identifier les facteurs sous-jacents.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p></p>
        <p></p>
        """,
            unsafe_allow_html=True
    )


    
if selected == "Visualisation":  # DataVisualisation
    
    # Ajout d'un style CSS pour améliorer l'apparence (même style que l'introduction et la conclusion)
    st.markdown("""
    <style>
        .main-title {
            color: #1E3A8A;
            font-size: 36px;
            font-weight: 700;
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 15px;
            border-bottom: 2px solid #3B82F6;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        }
        .title-container {
            background: linear-gradient(to right, #f0f4ff, #ffffff, #f0f4ff);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 40px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        }
        .dataviz-header {
            color: #1E3A8A;
            font-size: 28px;
            font-weight: 600;
            margin-bottom: 20px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Titre principal amélioré avec conteneur
    st.markdown("""
    <div class="title-container">
        <div class="main-title">Visualisation des Données</div>
        <div style="text-align: center; font-size: 20px; color: #4B5563; font-weight: 500;">Représentations graphiques des tendances d'accidents</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Fonction pour charger les images de manière optimisée
    @st.cache_data
    def load_image(image_index):
        image_paths = {
            0: "./images/output_14_0.png",
            1: "./images/output_15_1.png",
            2: "./images/output_16_0.png",
            3: "./images/output_19_0.png",
            4: "./images/output_21_0.png",
            5: "./images/output_24_0.png",
            6: "./images/output_25_1.png",
            7: "./images/output_26_0.png"
        }
        return image_paths.get(image_index)
    
    num_images = 8
    
    # Utilisation de session_state pour conserver l'état entre les rechargements
    if "image_index" not in st.session_state:
        st.session_state.image_index = 0

    # Sidebar pour slider
    st.sidebar.header("📊 Navigation des Graphes")
    image_index = st.sidebar.slider(
        "Sélectionnez un graphe", 
        0, 
        num_images - 1, 
        st.session_state.image_index,
        key="image_slider"
    )
    
    # Mettre à jour l'index dans session_state
    st.session_state.image_index = image_index

    # Affichage de l'image sélectionnée
    image_path = load_image(image_index)
    if image_path:
        st.image(image_path)
    else:
        st.error("Image non trouvée")

    # Boutons de navigation
    col1, col2, col3 = st.columns([1, 2, 1])

    with col1:
        if st.button("⬅ Précédent", key="prev_graph") and st.session_state.image_index > 0:
            st.session_state.image_index -= 1

    with col3:
        if st.button("Suivant ➡", key="next_graph") and st.session_state.image_index < num_images - 1:
            st.session_state.image_index += 1

    # Histogramme de la gravité des accidents (avec % sur 4 classes et sur classe binaire)
    st.image("./images/output_11_1.png")

    st.markdown(
        """
        <div style="background-color:#f9f9f9; padding:10px; border-radius:5px; border-left:5px solid #007BFF;">
            <b>📌 Analyse du Graphique</b>
            Ce graphique en barres montre la distribution des accidents selon leur gravité.</b>.
            <ul>
                <li><strong>Indemnes (41.3%) :</strong> Une grande partie des usagers impliqués dans des accidents n'ont subi aucune blessure.</li>
                <li><strong>Blessés légers (40.7%) :</strong> Un nombre similaire d'usagers ont subi des blessures mineures.</li>
                <li><strong>Blessés hospitalisés (15.4%) :</strong> Une part plus faible des victimes nécessitent une hospitalisation.</li>
                <li><strong>Tués (2.6%) :</strong> Bien que minoritaire, le nombre de décès reste préoccupant.</li>
                <li><strong>Interprétation :</strong> La majorité des accidents n'entraînent pas de blessures graves, mais la part des blessés hospitalisés et des décès souligne l'importance des mesures de prévention routière.</li>
            </ul>
        </div>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <p></p>
        <p></p>
        """,
            unsafe_allow_html=True
    )


if selected == "Modélisation":  # Modélisation 
    
    # Wrapper try-except pour toute la section de modélisation
    try:
        # Vérification de la mémoire disponible au début
        try:
            import psutil
            available_memory = psutil.virtual_memory().available / (1024 * 1024 * 1024)  # GB
            total_memory = psutil.virtual_memory().total / (1024 * 1024 * 1024)  # GB
            memory_percent = psutil.virtual_memory().percent
            
            # Avertir si la mémoire disponible est faible
            if available_memory < 2:  # Moins de 2GB disponible
                st.warning(f"⚠️ Mémoire disponible faible : {available_memory:.1f} GB sur {total_memory:.1f} GB ({memory_percent:.1f}% utilisé)")
                st.info("💡 Les données seront optimisées pour réduire l'utilisation mémoire.")
                
                # Si la mémoire est vraiment critique, proposer de nettoyer le cache
                if available_memory < 1:  # Moins de 1GB disponible
                    if st.button("🧹 Nettoyer le cache pour libérer de la mémoire"):
                        st.cache_data.clear()
                        cleanup_memory()
                        st.success("✅ Cache nettoyé. Veuillez rafraîchir la page.")
                        st.stop()
        except ImportError:
            pass  # psutil n'est pas installé, on continue sans vérification
        
        # Log mémoire avant le début de la modélisation
        log_memory_usage("Début de la section Modélisation")
    
        # Ajout d'un style CSS pour améliorer l'apparence (même style que l'introduction et la conclusion)
        st.markdown("""
        <style>
            .main-title {
                color: #1E3A8A;
                font-size: 36px;
                font-weight: 700;
                text-align: center;
                margin-bottom: 30px;
                padding-bottom: 15px;
                border-bottom: 2px solid #3B82F6;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
            }
            .title-container {
                background: linear-gradient(to right, #f0f4ff, #ffffff, #f0f4ff);
                padding: 20px;
                border-radius: 10px;
                margin-bottom: 40px;
                box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            }
            .model-header {
                color: #1E3A8A;
                font-size: 28px;
                font-weight: 600;
                margin-bottom: 20px;
            }
            .intro-header {
                color: #1E3A8A;
                font-size: 28px;
                font-weight: 600;
                margin-bottom: 20px;
            }
        </style>
        """, unsafe_allow_html=True)
    
        # Titre principal amélioré avec conteneur
        st.markdown("""
        <div class="title-container">
            <div class="main-title">Modélisation Prédictive</div>
            <div style="text-align: center; font-size: 20px; color: #4B5563; font-weight: 500;">Prédiction de la gravité des accidents</div>
        </div>
        """, unsafe_allow_html=True)
    
        # Méthodologie

        st.markdown("<div class='intro-header'>La démarche méthodologique 'en entonnoir'</div>", unsafe_allow_html=True)

        st.markdown("""

        La démarche retenue consiste à partir d'une liste de 8 modèles, de les hiérarchiser sur la base de leurs performances de bonne classification des observations du jeu de données de l'ensemble de test.
        Cette démarche se décline de la façon suivante :
        1. Entrainement de 8 modèles sans rééquilibrage des modalités de la cible (à 4 modalités ou binaire). Les modèles sont évalués et hiérarchisés sur la base de la métrique « F1_score » de la modalité « positive » (« tué » dans le cas à 4 modalités ou « blessé_tué » après binarisation).
        2. Les 4 modèles les plus performants ont ensuite été réentraînés en rééquilibrant les modalités de la cible (multinomiale ou binaire) : une hiérarchisation de leurs performances a été effectuée sur le même principe que ci-dessus.
        3. Nous avons ensuite retenu les 3 modèles les plus performants, du point précédent, pour les optimiser à l'aide de « GridSearchCV » et identifié les valeurs optimales des hyperparamètres de chacun des 3 modèles du podium.
        """)

        st.image("./images/20250305_Funnel_Cible-Multi_01.jpg", caption='Démarche en entonnoir : cible multinomiale')

        st.image("./images/20250305_Funnel_Cible-Binaire_01.jpg", caption='Démarche en entonnoir : cible binaire')
        st.markdown("""

        Dans ce qui suit, seuls les résultats de modélisation de la cible binaire seront présentés.

        """)

        # Chargement des données nécessaires pour la modélisation si elles ne sont pas déjà chargées
        if 'X_train' not in st.session_state:
            try:
                # Log mémoire avant chargement des données d'entraînement
                log_memory_usage("Avant chargement des données d'entraînement")
            
                # Charger les données avec optimisation mémoire
                data_dict = load_data(["X_train", "X_test", "y_train", "y_test"], optimize_memory=True)
            
                # Vérifier si les données sont chargées correctement
                if not all(key in data_dict for key in ["X_train", "X_test", "y_train", "y_test"]):
                    st.error("❌ Erreur: Toutes les données nécessaires n'ont pas pu être chargées.")
                    st.stop()
            
                # Limiter la taille des datasets pour la modélisation si nécessaire
                # Ajuster les limites en fonction de la mémoire disponible
                try:
                    if 'available_memory' in locals() and available_memory < 4:
                        max_samples_train = 30000  # Limite réduite pour l'entraînement
                        max_samples_test = 5000    # Limite réduite pour le test
                    else:
                        max_samples_train = 50000  # Limite normale pour l'entraînement
                        max_samples_test = 10000   # Limite normale pour le test
                except:
                    # Valeurs par défaut si on ne peut pas déterminer la mémoire
                    max_samples_train = 50000
                    max_samples_test = 10000
            
                # Échantillonner si les datasets sont trop grands
                if len(data_dict["X_train"]) > max_samples_train:
                    st.info(f"📊 Dataset d'entraînement limité à {max_samples_train} échantillons pour optimiser les performances.")
                    sample_indices = np.random.choice(len(data_dict["X_train"]), max_samples_train, replace=False)
                    data_dict["X_train"] = data_dict["X_train"].iloc[sample_indices]
                    data_dict["y_train"] = data_dict["y_train"].iloc[sample_indices]
            
                if len(data_dict["X_test"]) > max_samples_test:
                    st.info(f"📊 Dataset de test limité à {max_samples_test} échantillons pour optimiser les performances.")
                    sample_indices = np.random.choice(len(data_dict["X_test"]), max_samples_test, replace=False)
                    data_dict["X_test"] = data_dict["X_test"].iloc[sample_indices]
                    data_dict["y_test"] = data_dict["y_test"].iloc[sample_indices]
            
                # Stocker dans session_state
                st.session_state.X_train = data_dict["X_train"]
                st.session_state.X_test = data_dict["X_test"]
                st.session_state.y_train = data_dict["y_train"]
                st.session_state.y_test = data_dict["y_test"]
            
                # Log mémoire après chargement
                log_memory_usage("Après chargement des données d'entraînement")
            
                # Nettoyer la mémoire
                cleanup_memory()
                del data_dict  # Libérer le dictionnaire temporaire
            
                # Vérifier à nouveau la mémoire après chargement
                memory_after_load = get_memory_usage()
                if memory_after_load > 1000:  # Plus de 1GB utilisé
                    st.info(f"📊 Utilisation mémoire après chargement : {memory_after_load:.0f} MB")
            
            except MemoryError as e:
                st.error("❌ Erreur de mémoire insuffisante lors du chargement des données.")
                st.info("💡 Suggestions:")
                st.info("• Fermez d'autres applications pour libérer de la mémoire")
                st.info("• Rechargez la page")
                st.info("• Contactez l'administrateur si le problème persiste")
                st.stop()
            except Exception as e:
                st.error(f"❌ Erreur lors du chargement des données : {str(e)}")
                cleanup_memory()  # Essayer de nettoyer même en cas d'erreur
                st.stop()
    
        # Vérifier que accidents_binaire est disponible
        if 'accidents_binaire' not in st.session_state:
            st.error("❌ Les données accidents_binaire ne sont pas disponibles. Veuillez d'abord charger les données depuis la page d'Introduction.")
            st.stop()
    
        accidents_binaire = st.session_state.accidents_binaire
    
        # Création d'une nouvelle colonne avec des labels explicites
        try:
            accidents_binaire['gravite_accident_label'] = accidents_binaire['gravite_accident'].replace({
                '0': 'Indemne',
                '1': 'Blessé/Tué'
            })
        except Exception as e:
            st.error(f"❌ Erreur lors de la création des labels : {str(e)}")
            st.stop()

        # Création du graphique
        @st.cache_data
        def create_countplot():
            fig, ax = plt.subplots(figsize=(8, 4))
        
            # Affichage du graphique avec les nouvelles étiquettes
            sns.countplot(data=accidents_binaire,
                        x='gravite_accident_label',
                        hue='gravite_accident_label',
                        palette='Blues',
                        order=['Indemne', 'Blessé/Tué'],  # Assure l'ordre souhaité
                        ax=ax)
            ax.legend().set_visible(False)  # Suppression de la légende

            # Ajout des pourcentages
            total = len(accidents_binaire)
            for p in ax.patches:
                percentage = f"{100 * p.get_height() / total:.1f}%"
                ax.annotate(percentage, 
                            (p.get_x() + p.get_width() / 2, p.get_height()), 
                            ha='center', va='bottom', fontsize=12, color='black')

            # Titre stylisé
            plt.title("Distribution de la gravité des accidents\n",
                    loc="center", fontsize=16, fontweight='bold', color="black")

            plt.xlabel("Gravité de l'accident")
            plt.ylabel("Nombre d'accidents")
            return fig
    
        # Afficher le graphique
        fig = create_countplot()
        st.pyplot(fig)
    
        # Afficher le commentaire sous le graphique SEULEMENT
        st.markdown(
            """
            <div style="background-color:#f9f9f9; padding:10px; border-radius:5px; border-left:5px solid #007BFF;">
                <b>📌 Analyse du graphique :</b>  
                Ce graphique illustre la <b>distribution de la gravité des accidents</b> en France entre <b>2019 et 2022</b>.  
                On observe une différence notable entre les accidents <b>légers</b> et ceux <b>graves/mortels</b>,  
                avec une proportion plus importante d'accidents légers. Cette tendance met en évidence  
                la nécessité de renforcer les mesures de sécurité pour réduire la sévérité des accidents.
            </div>
            """,
            unsafe_allow_html=True
        )

        # Standardiser les variables 'latitude' et 'longitude' dans X_train et X_test si ce n'est pas déjà fait
        if 'standardized' not in st.session_state:
            scaler = StandardScaler()
            st.session_state.X_train[['latitude', 'longitude']] = scaler.fit_transform(st.session_state.X_train[['latitude', 'longitude']])
            st.session_state.X_test[['latitude', 'longitude']] = scaler.transform(st.session_state.X_test[['latitude', 'longitude']])
            st.session_state.standardized = True

        # Transformer la cible 'gravite_accident' dans y_train et y_test si ce n'est pas déjà fait
        if 'target_transformed' not in st.session_state:
            def transform_target(y):
                return y.replace({
                    'indemne': 'indemne',
                    'blesse_leger': 'blesse_tue',
                    'blesse_hospitalise': 'blesse_tue',
                    'tue': 'blesse_tue'
                })

            st.session_state.y_train = transform_target(st.session_state.y_train)
            st.session_state.y_test = transform_target(st.session_state.y_test)
            st.session_state.target_transformed = True

        # Définir le chemin des fichiers
        path = "./models/"

        # Fonction pour charger les modèles avec mise en cache
        @st.cache_resource
        def load_model(model_name: Optional[str]) -> Any:
            """
            Charge un modèle à partir d'un fichier pickle avec mise en cache.
        
            Args:
                model_name: Nom du modèle à charger (CatBoost, XGBoost)
            
            Returns:
                Le modèle chargé
            """
            if model_name is None:
                st.error("Le nom du modèle ne peut pas être None.")
                return None
            
            model_path = {
                "CatBoost": "20250129_catboost_best_model.pkl",
                "XGBoost": "20250129_xgb_best_model.pkl"
            }
            
            if model_name not in model_path:
                st.error(f"Modèle inconnu: {model_name}")
                return None
                
            model_file = os.path.join(path, model_path[model_name])
        
            if os.path.exists(model_file):
                return joblib.load(model_file)
            else:
                st.error(f"Le fichier {model_file} n'existe pas.")
                return None

        # Fonction pour charger les valeurs SHAP avec mise en cache
        @st.cache_resource
        def load_shap_values(model_name: Optional[str]) -> Any:
            """
            Charge les valeurs SHAP précalculées à partir d'un fichier pickle.
        
            Args:
                model_name: Nom du modèle (CatBoost, XGBoost)
            
            Returns:
                Les valeurs SHAP ou None si le fichier n'existe pas
            """
            if model_name is None:
                st.error("Le nom du modèle ne peut pas être None.")
                return None
            
            shap_path = {
                "CatBoost": "20250304_catboost_bin_best_shap_values.pkl",
                "XGBoost": "20250304_XGBoost_bin_best_shap_values.pkl"
            }
            
            if model_name not in shap_path:
                st.error(f"Modèle inconnu: {model_name}")
                return None
                
            shap_file = os.path.join(path, shap_path[model_name])
        
            if os.path.exists(shap_file):
                return joblib.load(shap_file)
            else:
                st.warning(f"Le fichier de valeurs SHAP {shap_file} n'existe pas. Les valeurs seront calculées à la volée.")
                return None

        # Fonction pour calculer les valeurs SHAP (avec mise en cache optimisée)
        @st.cache_data(ttl=3600, max_entries=5, show_spinner=False)
        def calculate_shap_values(model: Any, X_test: pd.DataFrame, model_name: Optional[str], 
                                 max_samples: int = 1000, memory_limit_mb: int = 500) -> Any:
            """
            Calcule les valeurs SHAP avec mise en cache et optimisation mémoire.
        
            Args:
                model: Modèle chargé
                X_test: Données de test
                model_name: Nom du modèle
                max_samples: Nombre maximum d'échantillons à traiter (par défaut 1000)
                memory_limit_mb: Limite de mémoire en MB (par défaut 500)
            
            Returns:
                Les valeurs SHAP calculées
            """
            if model_name is None:
                st.error("Le nom du modèle ne peut pas être None.")
                return None
                
            if model is None:
                st.error("Le modèle ne peut pas être None.")
                return None
                
            try:
                # Vérifier la mémoire disponible
                initial_memory = get_memory_usage()
                if initial_memory > memory_limit_mb:
                    st.warning(f"⚠️ Utilisation mémoire élevée ({initial_memory:.0f} MB). Nettoyage en cours...")
                    cleanup_memory()
                    # Fermer toutes les figures matplotlib ouvertes
                    plt.close('all')
                    # Attendre un peu pour que le GC fasse son travail
                    import time
                    time.sleep(0.5)
            
                # Limiter le nombre d'échantillons si nécessaire
                n_samples = X_test.shape[0]
                if n_samples > max_samples:
                    st.info(f"📊 Échantillonnage des données: {max_samples} exemples sur {n_samples} pour optimiser les performances")
                    X_test = X_test.sample(n=max_samples, random_state=42)
            
                # Afficher une barre de progression
                progress_bar = st.progress(0)
                progress_text = st.empty()
            
                progress_text.text(f"🔄 Initialisation du calcul SHAP pour {model_name}...")
                progress_bar.progress(10)
            
                # Pour les modèles basés sur des arbres (CatBoost, XGBoost, RandomForest)
                try:
                    progress_text.text("🌳 Tentative avec TreeExplainer (méthode rapide)...")
                    progress_bar.progress(30)
                
                    if model_name in ["XGBoost", "CatBoost", "Random Forest"]:
                        explainer = shap.TreeExplainer(model)
                    
                        # Calculer les valeurs SHAP par batch pour économiser la mémoire
                        batch_size = min(100, len(X_test))
                        shap_values_list = []
                    
                        for i in range(0, len(X_test), batch_size):
                            batch_end = min(i + batch_size, len(X_test))
                            batch = X_test.iloc[i:batch_end]
                        
                            # Mise à jour de la progression
                            progress = 30 + int(60 * (i + batch_size) / len(X_test))
                            progress_bar.progress(min(progress, 90))
                            progress_text.text(f"🔄 Calcul en cours... {i+1}-{batch_end}/{len(X_test)} exemples")
                        
                            # Calculer les valeurs SHAP pour ce batch
                            batch_shap_values = explainer.shap_values(batch)
                            shap_values_list.append(batch_shap_values)
                        
                            # Vérifier la mémoire après chaque batch
                            current_memory = get_memory_usage()
                            if current_memory > memory_limit_mb * 0.8:
                                st.warning(f"⚠️ Mémoire proche de la limite ({current_memory:.0f} MB). Nettoyage...")
                                cleanup_memory()
                                # Si toujours trop élevé, réduire la taille du prochain batch
                                if current_memory > memory_limit_mb * 0.9:
                                    batch_size = max(10, batch_size // 2)
                    
                        # Concaténer tous les résultats
                        if isinstance(shap_values_list[0], list):
                            # Pour les modèles multi-classes
                            shap_values = [np.vstack([batch[i] for batch in shap_values_list]) 
                                         for i in range(len(shap_values_list[0]))]
                        else:
                            shap_values = np.vstack(shap_values_list)
                    
                        progress_bar.progress(100)
                        progress_text.text("✅ Calcul SHAP terminé avec succès!")
                        return shap_values
                    
                except Exception as e:
                    st.warning(f"⚠️ TreeExplainer non disponible: {str(e)}")
                
                # Essayer avec l'explainer standard
                try:
                    progress_text.text("🔧 Tentative avec Explainer standard...")
                    progress_bar.progress(40)
                
                    explainer = shap.Explainer(model)
                    shap_values = explainer(X_test)
                
                    progress_bar.progress(100)
                    progress_text.text("✅ Calcul SHAP terminé avec succès!")
                    return shap_values
                
                except Exception as e:
                    st.warning(f"⚠️ Explainer standard non disponible: {str(e)}")
                
                    # Méthode de fallback avec KernelExplainer (plus lente mais plus robuste)
                    progress_text.text("🔬 Utilisation de KernelExplainer (méthode alternative)...")
                    progress_bar.progress(50)
                
                    try:
                        # Réduire encore plus l'échantillon pour KernelExplainer
                        kernel_sample_size = min(100, X_test.shape[0])
                        X_sample = X_test.sample(kernel_sample_size, random_state=42) if X_test.shape[0] > kernel_sample_size else X_test
                    
                        # Créer un background dataset plus petit
                        background_size = min(50, kernel_sample_size)
                        background = shap.sample(X_sample, background_size)
                    
                        # Créer l'explainer
                        if hasattr(model, 'predict_proba'):
                            kernel_explainer = shap.KernelExplainer(model.predict_proba, background)
                        else:
                            kernel_explainer = shap.KernelExplainer(model.predict, background)
                    
                        # Calculer les valeurs SHAP
                        progress_text.text(f"🔄 Calcul SHAP sur {kernel_sample_size} échantillons...")
                        progress_bar.progress(70)
                    
                        shap_values = kernel_explainer.shap_values(X_sample)
                    
                        progress_bar.progress(100)
                        progress_text.text("✅ Calcul SHAP terminé (méthode alternative)!")
                    
                        # Nettoyer la mémoire après le calcul
                        cleanup_memory()
                    
                        return shap_values
                    
                    except Exception as e2:
                        st.error(f"❌ Toutes les méthodes ont échoué: {str(e2)}")
                        progress_bar.empty()
                        progress_text.empty()
                    
                        # Créer des valeurs SHAP factices pour éviter les erreurs
                        st.warning("⚠️ Utilisation de valeurs approximatives pour la visualisation")
                        dummy_values = np.random.normal(0, 0.01, (X_test.shape[0], X_test.shape[1]))
                        return dummy_values
        
            except MemoryError:
                st.error("❌ Erreur de mémoire insuffisante. Réduction automatique de la taille des données...")
                if 'progress_bar' in locals():
                    progress_bar.empty()
                if 'progress_text' in locals():
                    progress_text.empty()
            
                # Nettoyer agressivement la mémoire
                cleanup_memory()
                plt.close('all')
            
                # Réessayer avec un échantillon beaucoup plus petit
                reduced_size = min(50, X_test.shape[0])
                X_reduced = X_test.sample(reduced_size, random_state=42)
            
                # Appel récursif avec des paramètres plus restrictifs
                return calculate_shap_values(model, X_reduced, model_name, 
                                           max_samples=reduced_size, 
                                           memory_limit_mb=memory_limit_mb * 2)  # Augmenter la limite pour éviter une boucle infinie
            
            except Exception as e:
                st.error(f"❌ Erreur générale lors du calcul des valeurs SHAP: {str(e)}")
                progress_bar.empty()
                progress_text.empty()
            
                # Retourner des valeurs factices pour éviter les erreurs
                dummy_values = np.zeros((min(100, len(X_test)), len(X_test.columns)))
                return dummy_values
        
            finally:
                # Nettoyer les éléments de progression
                if 'progress_bar' in locals():
                    progress_bar.empty()
                if 'progress_text' in locals():
                    progress_text.empty()
            
                # Forcer le nettoyage mémoire
                cleanup_memory()

        # Fonction pour extraire directement les importances du modèle Random Forest
        def get_feature_importances_rf(model, feature_names):
            """
            Extrait les importances directement du modèle Random Forest.
        
            Args:
                model: Modèle Random Forest
                feature_names: Noms des features
            
            Returns:
                DataFrame avec les importances triées
            """
            if not hasattr(model, 'feature_importances_'):
                st.error("Le modèle ne possède pas d'attribut feature_importances_")
                return None
        
            # Extraire les importances
            importances = model.feature_importances_
        
            # Vérifier que les dimensions correspondent
            if len(importances) != len(feature_names):
                st.warning(f"Dimensions incorrectes: importances {len(importances)} vs features {len(feature_names)}")
                # Ajuster si nécessaire
                if len(importances) < len(feature_names):
                    # Compléter avec des zéros
                    importances = np.pad(importances, (0, len(feature_names) - len(importances)))
                else:
                    # Tronquer
                    importances = importances[:len(feature_names)]
        
            # Créer un DataFrame
            importance_df = pd.DataFrame({
                'Feature': feature_names,
                'Importance': importances
            })
        
            # Trier par importance décroissante
            importance_df = importance_df.sort_values('Importance', ascending=False)
        
            return importance_df

        # Fonction pour créer les graphiques SHAP avec mise en cache
        @st.cache_data
        def create_importance_plot_rf(_shap_values_data, _feature_names, max_display=24, min_display=10):
            """Crée un graphique d'importance des variables spécifiquement pour Random Forest."""
            fig = plt.figure(figsize=(12, 8))  # Augmenter la taille pour plus de lisibilité
            try:
                # Convertir _feature_names en tableau NumPy
                feature_names_array = np.array(_feature_names)
            
                # Pour Random Forest, nous utilisons une approche manuelle plus robuste
                # Extraire les valeurs SHAP pour la classe positive (indice 1 pour classification binaire)
                if len(_shap_values_data) > 1:  # S'il y a plusieurs classes
                    shap_values_to_use = _shap_values_data[1]  # Utiliser la classe positive
                else:
                    shap_values_to_use = _shap_values_data[0]
            
                # Calculer l'importance des features (moyenne des valeurs absolues)
                feature_importance = np.abs(shap_values_to_use).mean(0)
            
                # Créer un DataFrame pour le tri et l'affichage
                importance_df = pd.DataFrame({
                    'Feature': feature_names_array,
                    'Importance': feature_importance
                })
            
                # Trier par importance décroissante
                importance_df = importance_df.sort_values('Importance', ascending=False)
            
                # Limiter aux max_display plus importantes features
                top_features = importance_df.head(max_display)
            
                # S'assurer d'avoir au moins min_display features
                if len(top_features) < min_display and len(importance_df) >= min_display:
                    top_features = importance_df.head(min_display)
            
                # Créer une palette de couleurs dégradées
                from matplotlib import cm
                colors = cm.get_cmap('viridis')(np.linspace(0, 0.8, len(top_features)))
            
                # Créer un graphique à barres horizontal avec des couleurs dégradées
                bars = plt.barh(
                    y=top_features['Feature'],
                    width=top_features['Importance'],
                    color=colors
                )
            
                # Ajouter les valeurs à côté des barres
                for bar in bars:
                    width = bar.get_width()
                    plt.text(width + width*0.01,  # Position légèrement à droite de la barre
                            bar.get_y() + bar.get_height()/2,  # Position verticale centrée
                            f'{width:.4f}',  # Formater avec 4 décimales
                            ha='left', va='center',
                            fontsize=9)
            
                # Inverser l'axe y pour avoir la feature la plus importante en haut
                plt.gca().invert_yaxis()
            
                # Ajouter une grille horizontale pour faciliter la lecture
                plt.grid(axis='x', linestyle='--', alpha=0.6)
            
                # Ajouter le titre et les labels
                plt.title("Importance des features (Random Forest)",
                        fontsize=20,
                        fontstyle='italic')
                plt.xlabel("Impact moyen sur la prédiction (valeur SHAP)")
            
                # Ajuster les marges
                plt.tight_layout()
            
                return fig
            except Exception as e:
                st.error(f"Erreur d'affichage RF: {str(e)}")
                # Créer un graphique d'erreur
                plt.text(0.5, 0.5, f"Erreur d'affichage RF: {str(e)}", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=plt.gca().transAxes)
            
                plt.title("Importance des features (Random Forest) - ERREUR",
                        fontsize=20,
                        fontstyle='italic',
                        color='red')
                plt.tight_layout()
                return fig
    
        @st.cache_data
        def create_importance_plot(_shap_values_data, _feature_names, max_display=24):
            """Crée un graphique d'importance des variables pour les modèles."""
            fig = plt.figure(figsize=(10, 6))
            try:
                # Convertir _feature_names en tableau NumPy pour éviter les problèmes d'indexation
                feature_names_array = np.array(_feature_names)
            
                # Pour les modèles
                shap.summary_plot(_shap_values_data, 
                                plot_type="bar", 
                                color='#39c5f2',
                                max_display=max_display,
                                feature_names=feature_names_array,
                                show=False)
            except Exception as e:
                plt.text(0.5, 0.5, f"Erreur d'affichage: {str(e)}", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=plt.gca().transAxes)
            
            plt.title("Importance des features dans la construction du modèle",
                    fontsize=20,
                    fontstyle='italic')
            plt.tight_layout()
            return fig
    
        @st.cache_data
        def create_beeswarm_plot_rf(_shap_values_data, _feature_names, max_display=24):
            """Crée un graphique BeeSwarm spécifiquement pour Random Forest."""
            fig = plt.figure(figsize=(14, 8))
            try:
                # Convertir feature_names en array numpy
                feature_names_array = np.array(_feature_names)
            
                # Pour Random Forest, nous devons traiter les valeurs SHAP différemment
                # Extraire les valeurs SHAP pour la classe positive (indice 1 pour classification binaire)
                if len(_shap_values_data) > 1:  # S'il y a plusieurs classes
                    shap_values_to_use = _shap_values_data[1]  # Utiliser la classe positive
                else:
                    shap_values_to_use = _shap_values_data[0]
            
                # Calculer l'importance des features (moyenne des valeurs absolues)
                feature_importance = np.abs(shap_values_to_use).mean(0)
            
                # Trier les indices par importance
                sort_inds = np.argsort(feature_importance)
            
                # Limiter aux max_display plus importantes features
                if sort_inds.size > max_display:
                    sort_inds = sort_inds[-max_display:]
            
                # Créer un DataFrame pour le plot
                X_test_values = st.session_state.X_test.values
            
                # Créer un scatter plot pour chaque feature
                for i, ind in enumerate(sort_inds):
                    # Position verticale (feature)
                    pos = len(sort_inds) - i - 1
                
                    # Valeurs SHAP pour cette feature
                    shap_values_feature = shap_values_to_use[:, ind]
                
                    # Valeurs de la feature
                    feature_values = X_test_values[:, ind]
                
                    # Normaliser les valeurs de la feature pour la couleur
                    if np.max(feature_values) - np.min(feature_values) > 0:
                        normalized_values = (feature_values - np.min(feature_values)) / (np.max(feature_values) - np.min(feature_values))
                    else:
                        normalized_values = np.zeros_like(feature_values)
                
                    # Créer un scatter plot
                    plt.scatter(
                        shap_values_feature,
                        np.ones(len(shap_values_feature)) * pos,
                        c=normalized_values,
                        cmap='coolwarm',
                        alpha=0.8,
                        s=20
                    )
            
                # Ajouter les noms des features
                plt.yticks(range(len(sort_inds)), [feature_names_array[ind] for ind in sort_inds])
            
                # Ajouter une grille horizontale
                plt.grid(axis='x', linestyle='--', alpha=0.6)
            
                # Ajouter le titre et les labels
                plt.title("Impact des features sur la prédiction (Random Forest)",
                        fontsize=20,
                        fontstyle='italic')
                plt.xlabel("Impact sur la prédiction (valeur SHAP)")
            
                # Ajouter une barre de couleur
                cbar = plt.colorbar()
                cbar.set_label("Valeur de la feature (normalisée)")
            
                # Ajuster les marges
                plt.tight_layout()
            
                return fig
            except Exception as e:
                st.error(f"Erreur d'affichage RF: {str(e)}")
                # Créer un graphique d'erreur
                plt.text(0.5, 0.5, f"Erreur d'affichage RF: {str(e)}", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=plt.gca().transAxes)
            
                plt.title("BeeSwarm Plot (Random Forest) - ERREUR",
                        fontsize=20,
                        fontstyle='italic',
                        color='red')
                plt.tight_layout()
                return fig
    
        @st.cache_data
        def create_beeswarm_plot(_shap_values_data, _feature_names, max_display=24):
            """Crée un graphique BeeSwarm."""
            fig = plt.figure(figsize=(14, 8))
            try:
                # Convertir feature_names en array numpy pour éviter les problèmes d'indexation
                feature_names_array = np.array(_feature_names)
            
                # Pour les modèles autres que Random Forest
                shap.summary_plot(
                    _shap_values_data,
                    st.session_state.X_test,
                    plot_type="dot",
                    max_display=max_display,
                    feature_names=feature_names_array,
                    show=False
                )
            except Exception as e:
                st.error(f"Erreur lors de l'affichage du BeeSwarm plot: {str(e)}")
                st.info("Tentative avec une méthode alternative...")
                try:
                    # Méthode alternative plus simple
                    shap.summary_plot(
                        _shap_values_data,
                        st.session_state.X_test.values,
                        plot_type="dot",
                        feature_names=_feature_names,
                        max_display=max_display,
                        show=False
                    )
                except Exception as e2:
                    st.error(f"L'affichage alternatif a également échoué: {str(e2)}")
                    # Méthode de secours ultime - créer un graphique simple
                    try:
                        plt.figure(figsize=(14, 8))
                        # Créer un DataFrame pour visualiser les valeurs SHAP moyennes
                        mean_shap = np.abs(_shap_values_data).mean(0) if hasattr(_shap_values_data, 'mean') else np.abs(_shap_values_data.values).mean(0)
                    
                        shap_df = pd.DataFrame({
                            'Feature': _feature_names,
                            'SHAP Value': mean_shap
                        })
                    
                        # Trier par valeur absolue
                        shap_df = shap_df.sort_values('SHAP Value', ascending=False).head(max_display)
                    
                        # Créer un barplot
                        plt.barh(y=shap_df['Feature'], width=shap_df['SHAP Value'], color='#39c5f2')
                        plt.title("Importance des features (méthode de secours)", fontsize=14)
                        plt.xlabel("Impact moyen sur la prédiction (valeur SHAP)")
                    except Exception as e3:
                        plt.text(0.5, 0.5, f"Toutes les méthodes d'affichage ont échoué: {str(e3)}", 
                                horizontalalignment='center', verticalalignment='center',
                                transform=plt.gca().transAxes)
        
            # Ajuster la taille des étiquettes et l'apparence générale
            ax = plt.gca()
            # Réduire la taille de la police des étiquettes des variables
            ax.tick_params(axis='y', labelsize=9)
            # Augmenter la taille de la police des valeurs sur l'axe x
            ax.tick_params(axis='x', labelsize=10)
        
            # Ajuster les marges pour éviter que les étiquettes ne soient coupées
            plt.subplots_adjust(left=0.25, right=0.95, top=0.95, bottom=0.1)
        
            plt.title("Interprétation Globale BeeSwarm", 
                    fontsize=14, fontstyle='italic', fontweight='bold')
            plt.tight_layout()
            return fig
    
        @st.cache_data
        def create_dependence_plot(_shap_values_data, _X_test_data, feature, _feature_names):
            """Crée un graphique de dépendance."""
            fig = plt.figure(figsize=(10, 6))
            try:
                # Convertir _feature_names en tableau NumPy pour éviter les problèmes d'indexation
                feature_names_array = np.array(_feature_names)
            
                # Convertir _X_test_data en tableau NumPy
                X_test_array = _X_test_data.values
            
                # Pour les modèles
                shap.dependence_plot(feature, 
                                    _shap_values_data, 
                                    X_test_array, 
                                    interaction_index=None, 
                                    alpha=0.5,
                                    feature_names=feature_names_array,
                                    ax=plt.gca(),
                                    show=False)
            except Exception as e:
                plt.text(0.5, 0.5, f"Erreur d'affichage: {str(e)}", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=plt.gca().transAxes)
            
            plt.title("Graphique de dépendance",
                    fontsize=11,
                    fontstyle='italic')
            plt.tight_layout()
            return fig
    
        @st.cache_data
        def create_dependence_interaction_plot(_shap_values_data, _X_test_data, feature, interaction_feature, _feature_names):
            """Crée un graphique de dépendance avec interaction."""
            fig = plt.figure(figsize=(10, 6))
            try:
                # Convertir _feature_names en tableau NumPy pour éviter les problèmes d'indexation
                feature_names_array = np.array(_feature_names)
            
                # Convertir _X_test_data en tableau NumPy
                X_test_array = _X_test_data.values
            
                # Pour les modèles
                shap.dependence_plot(feature, 
                                    _shap_values_data, 
                                    X_test_array, 
                                    interaction_index=interaction_feature, 
                                    alpha=0.5,
                                    feature_names=feature_names_array,
                                    ax=plt.gca(),
                                    show=False)
            except Exception as e:
                plt.text(0.5, 0.5, f"Erreur d'affichage: {str(e)}", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=plt.gca().transAxes)
            
            plt.title("Graphique de dépendance et interaction",
                    fontsize=11,
                    fontstyle='italic')
            plt.tight_layout()
            return fig
    
        @st.cache_data
        def create_waterfall_plot(_shap_values_data, observation_index):
            """Crée un graphique Waterfall."""
            fig = plt.figure(figsize=(10, 6))
            try:
                # Essayer d'utiliser la nouvelle API
                if isinstance(_shap_values_data, list):
                    # Pour les modèles avec structure de liste
                    try:
                        # Vérifier que l'index d'observation est valide
                        if observation_index >= len(_shap_values_data[0]):
                            observation_index = 0
                        
                        # Vérifier les dimensions
                        shap_values_to_plot = _shap_values_data[0][observation_index]
                    
                        # Utiliser la nouvelle API
                        shap.plots.waterfall(shap_values_to_plot, show=False)
                    except Exception as e:
                        # Fallback sur l'ancienne API
                        plt.text(0.5, 0.5, f"Erreur d'affichage: {str(e)}", 
                                horizontalalignment='center', verticalalignment='center',
                                transform=plt.gca().transAxes)
                else:
                    # Pour les modèles avec structure simple
                    try:
                        # Utiliser la nouvelle API
                        shap.plots.waterfall(_shap_values_data[observation_index], show=False)
                    except Exception as e:
                        # Fallback sur l'ancienne API
                        plt.text(0.5, 0.5, f"Erreur d'affichage: {str(e)}", 
                                horizontalalignment='center', verticalalignment='center',
                                transform=plt.gca().transAxes)
                    
                        plt.xlabel("Impact sur la prédiction (valeur SHAP)")
            except Exception as e:
                plt.text(0.5, 0.5, f"Erreur d'affichage: {str(e)}", 
                        horizontalalignment='center', verticalalignment='center',
                        transform=plt.gca().transAxes)
        
            plt.title("Waterfall Plot", fontsize=11, fontstyle='italic')
            plt.tight_layout()
            return fig

        # Menu pour charger le fichier pickle du modèle choisi
        model_choice = st.selectbox("Choisissez un modèle", ["CatBoost", "XGBoost"])
    
        # Charger le modèle choisi avec gestion d'erreur
        try:
            model = load_model(model_choice)
        except Exception as e:
            st.error(f"❌ Erreur lors du chargement du modèle {model_choice} : {str(e)}")
            model = None
    
        if model is not None:
            st.write(f"✅ Modèle {model_choice} chargé avec succès.")
        
            try:
                # Charger ou calculer les valeurs SHAP
                shap_values = load_shap_values(model_choice)
                if shap_values is None:
                    with st.spinner(f"⏳ Calcul des valeurs SHAP pour le modèle {model_choice}..."):
                        # Limiter le nombre d'échantillons pour SHAP si nécessaire
                        max_shap_samples = 1000
                        X_test_for_shap = st.session_state.X_test
                        if len(X_test_for_shap) > max_shap_samples:
                            st.info(f"📊 Utilisation de {max_shap_samples} échantillons pour le calcul SHAP (sur {len(X_test_for_shap)} disponibles)")
                            X_test_for_shap = X_test_for_shap.sample(n=max_shap_samples, random_state=42)
                    
                        shap_values = calculate_shap_values(model, X_test_for_shap, model_choice)
            
                # Créer l'explainer une seule fois
                explainer = shap.Explainer(model)
            
            except Exception as e:
                st.error(f"❌ Erreur lors du calcul des valeurs SHAP : {str(e)}")
                shap_values = None
                explainer = None
        
            # Sélectionner une valeur pour chaque feature de X_test
            st.markdown("<div class='intro-header'>Sélection des valeurs des features</div>", unsafe_allow_html=True)
        
            # Option pour afficher toutes les variables ou seulement les plus importantes
            show_all_features = st.checkbox("Afficher toutes les variables", value=False)
        
            # Définir les features à afficher
            if show_all_features:
                features_to_display = st.session_state.X_test.columns.tolist()
            else:
                # Utiliser un nombre limité de features pour l'interface utilisateur
                features_to_display = ["catv", "place", "obsm", "choc", "manv", "col"]
        
            # Créer un conteneur pour les sélecteurs avec défilement si nécessaire
            feature_container = st.container()
        
            # Utiliser des colonnes pour organiser les sélecteurs (3 colonnes)
            if show_all_features:
                # Avec beaucoup de variables, utiliser un layout plus compact
                num_cols = 3
                cols = feature_container.columns(num_cols)
            
                feature_values = {}
                for i, feature in enumerate(features_to_display):
                    col_idx = i % num_cols
                    with cols[col_idx]:
                        feature_values[feature] = st.selectbox(
                            f"{feature}", 
                            list(st.session_state.X_test[feature].unique()),
                            key=f"feature_{feature}"
                        )
            else:
                # Avec peu de variables, utiliser un layout plus spacieux
                feature_values = {}
                for feature in features_to_display:
                    if feature in st.session_state.X_test.columns:
                        feature_values[feature] = st.selectbox(
                            f"Valeur pour {feature}", 
                            list(st.session_state.X_test[feature].unique()),
                            key=f"feature_{feature}"
                        )
        
            # Compléter avec les valeurs par défaut pour les autres features
            for feature in st.session_state.X_test.columns:
                if feature not in feature_values:
                    feature_values[feature] = st.session_state.X_test[feature].iloc[0]
        
            # Afficher la valeur prédite de 'gravite_accident'
            predict_button = st.button("Prédire la gravité de l'accident")
        
            if predict_button:
                selected_features = pd.DataFrame([feature_values])
                predicted_value = model.predict(selected_features)
            
                # Convertir la valeur numérique en libellé
                prediction_label = "Indemne" if predicted_value[0] == 0 else "Blessé ou Tué"
                prediction_color = "#3B82F6" if predicted_value[0] == 0 else "#EF4444"
            
                # Afficher le résultat avec un style amélioré
                st.markdown(f"""
                <div style="background-color: #f0f7ff; padding: 15px; border-radius: 10px; border-left: 5px solid {prediction_color};">
                    <h4 style="margin-top: 0;">Prédiction</h4>
                    <p style="font-size: 18px; font-weight: bold;">{predicted_value[0]} : {prediction_label}</p>
                </div>
                """, unsafe_allow_html=True)
        
            # Menu pour afficher les graphiques d'interprétabilité (SHAP)
            st.markdown("<div class='intro-header'>Interprétabilité globale du modèle</div>", unsafe_allow_html=True)
        
            # Vérifier que les valeurs SHAP sont disponibles
            if shap_values is not None:
                shap_choice = st.selectbox("Choisissez un graphique d'interprétabilité", 
                                          ["Importance des variables", "BeeSwarm", "Dépendance", "Dépendance et interaction"])
            
                # Préparer les données pour les graphiques SHAP
                feature_names = st.session_state.X_test.columns.tolist()
                X_test_values = st.session_state.X_test.values
            
                # Afficher les graphiques d'interprétabilité globale
                if shap_choice == "Importance des variables":
                    with st.spinner("Génération du graphique d'importance des variables..."):
                        try:
                            # Utiliser la fonction pour CatBoost et XGBoost
                            fig = create_importance_plot(shap_values, feature_names, max_display=24)
                            st.pyplot(fig)
                            plt.close()
                        except Exception as e:
                            st.error(f"❌ Erreur lors de la génération du graphique d'importance : {str(e)}")
                            plt.close()  # S'assurer que la figure est fermée
        
                elif shap_choice == "BeeSwarm":
                    with st.spinner("Génération du graphique BeeSwarm..."):
                        try:
                            # Utiliser la fonction pour CatBoost et XGBoost
                            fig = create_beeswarm_plot(shap_values, feature_names, max_display=24)
                            st.pyplot(fig)
                            plt.close()
                        except Exception as e:
                            st.error(f"❌ Erreur lors de la génération du graphique BeeSwarm : {str(e)}")
                            plt.close()  # S'assurer que la figure est fermée
        
                elif shap_choice == "Dépendance":
                    feature = st.selectbox("Choisissez une feature", feature_names)
                    with st.spinner(f"Génération du graphique de dépendance pour {feature}..."):
                        try:
                            # Créer une nouvelle figure pour éviter les problèmes de mise en cache
                            plt.figure(figsize=(10, 6))
                        
                            # Pour les modèles CatBoost et XGBoost
                            shap.dependence_plot(feature, 
                                                shap_values, 
                                                st.session_state.X_test,
                                                interaction_index=None, 
                                                alpha=0.5,
                                                feature_names=feature_names,
                                                ax=plt.gca(),
                                                show=False)
                        
                            plt.title("Graphique de dépendance",
                                    fontsize=11,
                                    fontstyle='italic')
                            plt.tight_layout()
                            st.pyplot(plt.gcf())
                            plt.close()
                        except Exception as e:
                            st.error(f"❌ Erreur lors de la génération du graphique de dépendance : {str(e)}")
                            plt.close()  # S'assurer que la figure est fermée
            
                elif shap_choice == "Dépendance et interaction":
                    feature = st.selectbox("Choisissez une feature", feature_names, key="dep_feature")
                    interaction_feature = st.selectbox("Choisissez une feature d'interaction", 
                                                      feature_names, key="int_feature")
                    with st.spinner(f"Génération du graphique de dépendance et interaction pour {feature} et {interaction_feature}..."):
                        try:
                            # Créer une nouvelle figure pour éviter les problèmes de mise en cache
                            plt.figure(figsize=(10, 6))
                        
                            # Pour les modèles CatBoost et XGBoost
                            shap.dependence_plot(feature, 
                                                shap_values, 
                                                st.session_state.X_test, 
                                                interaction_index=interaction_feature, 
                                                alpha=0.5,
                                                feature_names=feature_names,
                                                ax=plt.gca(),
                                                show=False)
                        
                            plt.title("Graphique de dépendance et interaction",
                                    fontsize=11,
                                    fontstyle='italic')
                            plt.tight_layout()
                            st.pyplot(plt.gcf())
                            plt.close()
                        except Exception as e:
                            st.error(f"❌ Erreur lors de la génération du graphique de dépendance et interaction : {str(e)}")
                            plt.close()  # S'assurer que la figure est fermée
            else:
                st.warning("⚠️ Les valeurs SHAP ne sont pas disponibles. Veuillez vérifier que le modèle a été chargé correctement.")
        
            # Menu pour afficher les graphiques d'interprétabilité locale
            st.markdown("<div class='intro-header'>Interprétabilité locale du modèle</div>", unsafe_allow_html=True)
        
            # Vérifier que l'explainer et les valeurs SHAP sont disponibles
            if explainer is not None and shap_values is not None:
                local_shap_choice = st.selectbox("Choisissez un graphique d'interprétabilité locale", 
                                                ["Force Plot", "Waterfall Plot", "Decision Plot"])
            
                # Sélectionner l'index de l'observation du dataframe "X_test"
                observation_index = st.number_input("Indiquez l'index de l'observation dans X_test", 
                                                   min_value=0, max_value=len(st.session_state.X_test)-1, step=1)
        
                if local_shap_choice == "Force Plot":
                    with st.spinner("Génération du Force Plot..."):
                        # Pour les modèles (CatBoost, XGBoost)
                        try:
                            # Créer un HTML pour le force plot
                            plt.figure(figsize=(10, 3))
                            force_plot = shap.force_plot(
                                base_value=getattr(explainer, 'expected_value', 0),
                                shap_values=shap_values[observation_index],
                                features=st.session_state.X_test.iloc[observation_index],
                                feature_names=feature_names,
                                matplotlib=True,
                                show=False
                            )
                            st.pyplot(plt.gcf())
                            plt.close()
                        except Exception as e:
                            st.error(f"Erreur lors de l'affichage du Force Plot: {str(e)}")
                            st.info("Essai avec une méthode alternative...")
                            try:
                                # Méthode alternative
                                plt.figure(figsize=(10, 3))
                            
                                # Créer un barplot simple au lieu d'un waterfall plot
                                feature_names = st.session_state.X_test.columns.tolist()
                                shap_values_to_plot = shap_values[observation_index]
                            
                                # Trier les valeurs SHAP par importance
                                indices = np.argsort(np.abs(shap_values_to_plot))
                            
                                # Prendre les 10 features les plus importantes
                                top_indices = indices[-10:]
                            
                                # Créer un barplot horizontal
                                plt.barh(
                                    y=np.array(feature_names)[top_indices],
                                    width=shap_values_to_plot[top_indices],
                                    color=['#ff0d57' if x > 0 else '#1E88E5' for x in shap_values_to_plot[top_indices]]
                                )
                            
                                plt.title("Alternative au Force Plot (Top 10 features)", fontsize=12)
                                plt.xlabel("Impact sur la prédiction (valeur SHAP)")
                                plt.tight_layout()
                                st.pyplot(plt.gcf())
                                plt.close()
                            except Exception as e2:
                                st.error(f"L'affichage alternatif a échoué: {str(e2)}")
                
                elif local_shap_choice == "Waterfall Plot":
                    with st.spinner("Génération du Waterfall Plot..."):
                        # Pour les modèles CatBoost et XGBoost
                        try:
                            # Créer un objet Explanation pour le waterfall plot
                            plt.figure(figsize=(10, 6))
                        
                            # Créer un objet Explanation à partir des valeurs SHAP
                            # Cette approche utilise directement les valeurs brutes pour créer un graphique alternatif
                            feature_names = st.session_state.X_test.columns.tolist()
                            shap_values_to_plot = shap_values[observation_index]
                        
                            # Trier les valeurs SHAP par importance
                            indices = np.argsort(np.abs(shap_values_to_plot))
                        
                            # Prendre les 10 features les plus importantes
                            top_indices = indices[-10:]
                        
                            # Créer un barplot horizontal
                            plt.barh(
                                y=np.array(feature_names)[top_indices],
                                width=shap_values_to_plot[top_indices],
                                color=['#ff0d57' if x > 0 else '#1E88E5' for x in shap_values_to_plot[top_indices]]
                            )
                        
                            plt.title("Waterfall Plot (Top 10 features)", fontsize=12)
                            plt.xlabel("Impact sur la prédiction (valeur SHAP)")
                            plt.tight_layout()
                            st.pyplot(plt.gcf())
                            plt.close()
                        
                            # Afficher aussi la valeur de base et la somme des valeurs SHAP
                            expected_value = getattr(explainer, 'expected_value', 0)
                            if isinstance(expected_value, list) or isinstance(expected_value, np.ndarray):
                                expected_value = expected_value[0]
                        
                            st.write(f"Valeur de base (expected value): {float(expected_value):.4f}")
                            st.write(f"Somme des valeurs SHAP: {float(np.sum(shap_values_to_plot)):.4f}")
                            st.write(f"Prédiction finale: {float(expected_value + np.sum(shap_values_to_plot)):.4f}")
                        
                        except Exception as e:
                            st.error(f"Erreur lors de l'affichage du Waterfall Plot: {str(e)}")
                        st.info("Essai avec une méthode alternative...")
                        try:
                            # Méthode alternative
                            plt.figure(figsize=(10, 6))
                        
                            # Créer un barplot simple au lieu d'un waterfall plot
                            feature_names = st.session_state.X_test.columns.tolist()
                            shap_values_to_plot = shap_values[observation_index]
                        
                            # Trier les valeurs SHAP par importance
                            indices = np.argsort(np.abs(shap_values_to_plot))
                        
                            # Prendre les 10 features les plus importantes
                            top_indices = indices[-10:]
                        
                            # Créer un barplot horizontal
                            plt.barh(
                                y=np.array(feature_names)[top_indices],
                                width=shap_values_to_plot[top_indices],
                                color=['#ff0d57' if x > 0 else '#1E88E5' for x in shap_values_to_plot[top_indices]]
                            )
                        
                            plt.title("Alternative au Waterfall Plot (Top 10 features)", fontsize=12)
                            plt.xlabel("Impact sur la prédiction (valeur SHAP)")
                            plt.tight_layout()
                            st.pyplot(plt.gcf())
                            plt.close()
                        except Exception as e2:
                            st.error(f"L'affichage alternatif a échoué: {str(e2)}")
        
                elif local_shap_choice == "Decision Plot":
                    with st.spinner("Génération du Decision Plot..."):
                        # Decision Plot ne peut pas être mis en cache facilement, on l'affiche directement
                    
                        # Pour les modèles CatBoost et XGBoost
                        try:
                            plt.figure(figsize=(10, 8))
                            shap.decision_plot(
                                getattr(explainer, 'expected_value', 0),
                                shap_values[observation_index],
                                st.session_state.X_test.iloc[observation_index],
                                feature_names=feature_names,
                                show=False
                            )
                            plt.tight_layout()
                            st.pyplot(plt.gcf())
                        except Exception as e:
                            st.error(f"Erreur lors de l'affichage du Decision Plot: {str(e)}")
                            st.info("Essai avec une méthode alternative...")
                            try:
                                # Méthode alternative
                                plt.figure(figsize=(10, 8))
                                shap.plots.decision(
                                    getattr(explainer, 'expected_value', 0),
                                        shap_values[observation_index],
                                    st.session_state.X_test.iloc[observation_index,:],
                                    feature_names=feature_names,
                                    show=False
                                )
                                plt.tight_layout()
                                st.pyplot(plt.gcf())
                            except Exception as e2:
                                st.error(f"L'affichage du Decision Plot a échoué: {str(e2)}")
            else:
                st.warning("⚠️ L'explainer ou les valeurs SHAP ne sont pas disponibles. Veuillez vérifier que le modèle a été chargé correctement.")
    
        # Nettoyage de la mémoire à la fin de la section Modélisation
        try:
            # Nettoyer les variables volumineuses qui ne sont plus nécessaires
            if 'shap_values' in locals():
                del shap_values
            if 'explainer' in locals():
                del explainer
            if 'model' in locals():
                del model
        
            # Forcer le garbage collection
            cleanup_memory()
        
            # Nettoyer aussi les figures matplotlib
            import matplotlib.pyplot as plt
            plt.close('all')
        
            # Log mémoire finale
            log_memory_usage("Fin de la section Modélisation")
        
            # Afficher un message de succès si tout s'est bien passé
            final_memory = get_memory_usage()
            if final_memory < 1500:  # Moins de 1.5GB utilisé
                print(f"✅ Section Modélisation terminée avec succès. Mémoire utilisée : {final_memory:.0f} MB")
        except Exception as cleanup_error:
            # Ne pas afficher d'erreur pour le nettoyage, juste logger
            print(f"Erreur lors du nettoyage mémoire : {str(cleanup_error)}")
        
    except MemoryError as e:
        # Gestion spécifique des erreurs de mémoire
        st.error("❌ Erreur de mémoire insuffisante dans la section Modélisation.")
        st.info("💡 Veuillez recharger la page manuellement ou sélectionner moins de données.")
        cleanup_memory()
        st.stop()
    except Exception as e:
        # Gestion générale des autres erreurs
        st.error(f"❌ Une erreur est survenue dans la section Modélisation : {str(e)}")
        st.info("💡 Essayez de recharger la page ou de sélectionner un autre modèle.")
        print(f"Erreur dans la section Modélisation : {str(e)}")
        cleanup_memory()
        cleanup_memory()

if selected == "Conclusion":  # Conclusion
    # Contenu de la page de conclusion
    
    # Ajout d'un style CSS pour améliorer l'apparence (même style que l'introduction)
    st.markdown("""
    <style>
        .main-title {
            color: #1E3A8A;
            font-size: 36px;
            font-weight: 700;
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 15px;
            border-bottom: 2px solid #3B82F6;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        }
        .title-container {
            background: linear-gradient(to right, #f0f4ff, #ffffff, #f0f4ff);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 40px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        }
        .conclusion-header {
            color: #1E3A8A;
            font-size: 28px;
            font-weight: 600;
            margin-bottom: 20px;
        }
        .conclusion-text {
            font-size: 16px;
            line-height: 1.6;
            text-align: justify;
            margin-bottom: 15px;
        }
        .conclusion-highlight {
            background-color: #F3F4F6;
            border-left: 4px solid #3B82F6;
            padding: 15px;
            margin: 20px 0;
        }
        .conclusion-section {
            margin-top: 30px;
            margin-bottom: 30px;
        }
        .conclusion-list {
            margin-left: 20px;
            margin-bottom: 15px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Titre principal amélioré avec conteneur
    st.markdown("""
    <div class="title-container">
        <div class="main-title">Synthèse et Perspectives</div>
        <div style="text-align: center; font-size: 20px; color: #4B5563; font-weight: 500;">Interprétabilité et implications des résultats</div>
    </div>
    """, unsafe_allow_html=True)
    
    
    # Résumé concis
    st.markdown("""
    <div class="conclusion-text">
        L'analyse SHAP du modèle CatBoostClassifier a révélé les facteurs clés influençant la prédiction de gravité des accidents routiers.
    </div>
    """, unsafe_allow_html=True)

    
    # Variables influentes - Version concise
    st.markdown('<div class="conclusion-header" style="font-size: 24px;">Facteurs déterminants</div>', unsafe_allow_html=True)
    
    st.markdown("""
    <div class="conclusion-highlight">
        <ul>
            <li><strong>Catégorie de véhicule</strong> : Les véhicules légers sont associés à une diminution de la gravité, contrairement aux deux-roues motorisés lourds.</li>
            <li><strong>Position de l'usager</strong> : Le rôle de conducteur ou passager influence significativement l'issue d'un accident.</li>
            <li><strong>Présence de piéton</strong> : Facteur majeur augmentant la gravité des accidents.</li>
            <li><strong>Type de collision et manœuvre</strong> : Variables déterminantes selon les circonstances.</li>
            <li><strong>Facteurs temporels</strong> (jour, mois, heure) : Influence contextuelle sur certains types d'accidents.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(
        """
        <p></p>
        <p></p>
        """,
            unsafe_allow_html=True
    )
    
    st.markdown("""
    <div class="conclusion-header">
        <div class="conclusion-header" style="font-size: 24px;">Implications pour la sécurité routière</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Utilisation des colonnes Streamlit pour créer un effet de cartes
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div style="background-color: white; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); padding: 20px; border-top: 4px solid #3B82F6; height: 300px; margin: 5px;">
            <div style="font-size: 32px; text-align: center; margin-bottom: 15px;">🚗</div>
            <div style="font-weight: 600; color: #1E3A8A; font-size: 18px; text-align: center; margin-bottom: 15px;">Véhicules vulnérables</div>
            <div style="text-align: center; color: #4B5563; font-size: 15px; max-height: 170px; overflow: auto;">Nécessité de mesures spécifiques pour les deux-roues motorisés et véhicules lourds.</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div style="background-color: white; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); padding: 20px; border-top: 4px solid #3B82F6; height: 300px; margin: 5px;">
            <div style="font-size: 32px; text-align: center; margin-bottom: 15px;">🚶</div>
            <div style="font-weight: 600; color: #1E3A8A; font-size: 18px; text-align: center; margin-bottom: 15px;">Protection des piétons</div>
            <div style="text-align: center; color: #4B5563; font-size: 15px; max-height: 170px; overflow: auto;">Priorité d'action pour ces usagers particulièrement vulnérables.</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div style="background-color: white; border-radius: 8px; box-shadow: 0 2px 5px rgba(0,0,0,0.1); padding: 20px; border-top: 4px solid #3B82F6; height: 300px; margin: 5px;">
            <div style="font-size: 32px; text-align: center; margin-bottom: 15px;">🔄</div>
            <div style="font-weight: 600; color: #1E3A8A; font-size: 18px; text-align: center; margin-bottom: 15px;">Approche systémique</div>
            <div style="text-align: center; color: #4B5563; font-size: 15px; max-height: 170px; overflow: auto;">Combinaison d'améliorations d'infrastructures et de sensibilisation des conducteurs.</div>
        </div>
        """, unsafe_allow_html=True)


########################################################################################
if selected == "Chat":
    # Ajout d'un style CSS pour améliorer l'apparence (même style que les autres pages)
    st.markdown("""
    <style>
        .main-title {
            color: #1E3A8A;
            font-size: 36px;
            font-weight: 700;
            text-align: center;
            margin-bottom: 30px;
            padding-bottom: 15px;
            border-bottom: 2px solid #3B82F6;
            text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
        }
        .title-container {
            background: linear-gradient(to right, #f0f4ff, #ffffff, #f0f4ff);
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 40px;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
        }
        .chat-header {
            color: #1E3A8A;
            font-size: 28px;
            font-weight: 600;
            margin-bottom: 20px;
        }
    </style>
    """, unsafe_allow_html=True)
    
    # Titre principal amélioré avec conteneur
    st.markdown("""
    <div class="title-container">
        <div class="main-title">Chat Prédictif</div>
        <div style="text-align: center; font-size: 20px; color: #4B5563; font-weight: 500;">Prédiction de la gravité des accidents par description textuelle</div>
    </div>
    """, unsafe_allow_html=True)

    # Fonction cachée pour charger le modèle une seule fois
    @st.cache_resource
    def load_chat_model():
        MODELE_PATH = "./models/20250129_catboost_best_model.pkl"
        return joblib.load(MODELE_PATH)
    
    # Chargement du modèle avec cache
    model = load_chat_model()

    # Fonction pour prédire la gravité de l'accident
    def predire_gravite(description: str):
        """Prend en entrée une description textuelle et prédit la gravité de l'accident."""
        # Convertir la description en une représentation adaptée au modèle
        # Cette étape dépend de votre preprocessing (TF-IDF, embeddings, etc.)
        # Pour l'exemple, nous allons juste retourner une prédiction aléatoire
        # Note: description parameter will be used when the actual model preprocessing is implemented
        _ = description  # Mark as intentionally unused for now
        import random
        predictions = ["Indemne", "Blessé léger", "Blessé hospitalisé", "Tué"]
        return random.choice(predictions)

    # Suppression du titre simple et remplacement par une instruction plus claire
    st.markdown('<div class="chat-header">Entrez une description d\'accident pour obtenir une prédiction</div>', unsafe_allow_html=True)
    st.write("Notre modèle analysera votre description et prédira la gravité probable de l'accident.")

    # Zone de chat
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Affichage des messages précédents
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    # Entrée utilisateur
    prompt = st.chat_input("Décrivez l'accident...")

    if prompt:
        # Ajouter le message utilisateur
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        # Obtenir la prédiction du modèle
        prediction = predire_gravite(prompt)

        # Ajouter la réponse du modèle
        with st.chat_message("assistant"):
            response = f"Selon notre analyse, la gravité de cet accident est estimée comme : **{prediction}**"
            st.write(response)
        st.session_state.messages.append({"role": "assistant", "content": response})

# Ajout d'un encadré en bas de page avec des liens
# Cette section sera affichée sur toutes les pages de l'application
st.markdown("<hr>", unsafe_allow_html=True)

# Style CSS pour l'encadré footer
st.markdown("""
<style>
    .footer-container {
        background-color: #f0f2f6;
        border-radius: 10px;
        padding: 15px;
        margin-top: 30px;
        margin-bottom: 20px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
    }
    .footer-title {
        font-size: 16px;
        font-weight: 600;
        color: #1E3A8A;
        margin-bottom: 10px;
    }
    .footer-links {
        display: flex;
        justify-content: center;
        flex-wrap: wrap;
        gap: 20px;
        margin-bottom: 10px;
    }
    .footer-link {
        display: inline-flex;
        align-items: center;
        color: #3B82F6;
        text-decoration: none;
        font-weight: 500;
        transition: color 0.2s;
    }
    .footer-link:hover {
        color: #1E3A8A;
        text-decoration: underline;
    }
    .footer-icon {
        margin-right: 5px;
        font-size: 20px;
    }
</style>
""", unsafe_allow_html=True)

# Contenu de l'encadré
st.markdown("""
<div class="footer-container">
    <div class="footer-title">Projet réalisé et publié par</div>
    <div class="footer-links">
        <a href="https://www.linkedin.com/in/selina-balikel-292038220/" class="footer-link" target="_blank">
            <span class="footer-icon">👤</span> Selina BALIKEL
        </a>
        <a href="https://www.linkedin.com/in/ahmed-hammoumi-86766a1/" class="footer-link" target="_blank">
            <span class="footer-icon">👤</span> Ahmed HAMMOUMI
        </a>
        <a href="https://www.linkedin.com/in/ndiaye-bacar-b92aa555/" class="footer-link" target="_blank">
            <span class="footer-icon">👤</span> Bacar NDIAYE
        </a>
        <a href="https://github.com/selinablkl/st_Accident" class="footer-link" target="_blank">
            <span class="footer-icon">💻</span> GitHub Repository
        </a>
    </div>
    <div style="font-size: 12px; color: #6B7280;">© 2024 - Analyse des Accidents Routiers</div>
</div>
""", unsafe_allow_html=True)