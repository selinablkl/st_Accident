import gc
import psutil
import streamlit as st
import pandas as pd
import numpy as np
from typing import Optional, Tuple

def get_memory_usage() -> float:
    """Retourne l'utilisation mémoire en MB"""
    process = psutil.Process()
    return process.memory_info().rss / 1024 / 1024

def check_memory_limit() -> bool:
    """Vérifie si on approche de la limite mémoire"""
    memory_mb = get_memory_usage()
    # Limite à 500MB pour Streamlit Cloud
    return memory_mb < 500

def sample_data_if_needed(df: pd.DataFrame, max_rows: int = 10000) -> pd.DataFrame:
    """Échantillonne les données si elles sont trop grandes"""
    if len(df) > max_rows:
        st.warning(f"Dataset trop grand ({len(df)} lignes). Échantillonnage à {max_rows} lignes pour éviter les problèmes mémoire.")
        return df.sample(n=max_rows, random_state=42)
    return df

def optimize_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Optimise les types de données pour réduire l'usage mémoire"""
    for col in df.columns:
        col_type = df[col].dtype
        
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
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
    
    return df

def cleanup_memory():
    """Force le nettoyage de la mémoire"""
    gc.collect()
    
def safe_cache_clear():
    """Nettoie le cache Streamlit de manière sécurisée"""
    try:
        st.cache_data.clear()
    except:
        pass

@st.cache_data(ttl=3600)  # Cache pour 1 heure
def load_data_with_sampling(file_path: str, max_rows: Optional[int] = None) -> pd.DataFrame:
    """Charge les données avec échantillonnage si nécessaire"""
    try:
        # Première lecture pour compter les lignes
        total_rows = sum(1 for _ in open(file_path)) - 1
        
        if max_rows and total_rows > max_rows:
            # Échantillonnage aléatoire lors de la lecture
            skip_prob = 1 - (max_rows / total_rows)
            df = pd.read_csv(
                file_path,
                skiprows=lambda i: i > 0 and np.random.random() > skip_prob,
                encoding='utf-8'
            )
            df = df.head(max_rows)  # S'assurer qu'on ne dépasse pas la limite
        else:
            df = pd.read_csv(file_path, encoding='utf-8')
        
        # Optimiser les types de données
        df = optimize_dataframe(df)
        
        return df
    except Exception as e:
        st.error(f"Erreur lors du chargement: {str(e)}")
        return pd.DataFrame()

def monitor_memory_decorator(func):
    """Décorateur pour surveiller l'usage mémoire"""
    def wrapper(*args, **kwargs):
        start_memory = get_memory_usage()
        
        try:
            result = func(*args, **kwargs)
            
            end_memory = get_memory_usage()
            memory_increase = end_memory - start_memory
            
            if memory_increase > 100:  # Plus de 100MB d'augmentation
                st.warning(f"⚠️ Augmentation mémoire importante: +{memory_increase:.1f}MB")
                cleanup_memory()
            
            return result
        except MemoryError:
            st.error("❌ Erreur mémoire! L'opération nécessite trop de ressources.")
            cleanup_memory()
            safe_cache_clear()
            st.stop()
        except Exception as e:
            st.error(f"❌ Erreur: {str(e)}")
            raise
    
    return wrapper