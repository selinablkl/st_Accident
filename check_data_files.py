#!/usr/bin/env python3
"""
Script pour vérifier la présence des fichiers de données nécessaires.
"""

import os
import sys

# Définir les fichiers requis et leurs tailles approximatives
REQUIRED_FILES = {
    'data/20241031_accidents_Fr_19_22_AH_q.csv': '215MB',
    'data/X_train.csv': '100MB',
    'data/X_test.csv': '43MB',
    'data/y_train.csv': '3.9MB',
    'data/y_test.csv': '1.7MB',
    'data/departements-france.csv': '3.3KB',
    'models/20250129_catboost_best_model.pkl': 'Variable',
    'models/20250129_xgb_best_model.pkl': 'Variable',
    'models/20250304_XGBoost_bin_best_shap_values.pkl': 'Variable',
    'models/20250304_catboost_bin_best_shap_values.pkl': 'Variable',
    'communes-20220101-shp/communes-20220101.shp': 'Variable'
}

def check_files():
    """Vérifie la présence des fichiers requis."""
    missing_files = []
    existing_files = []
    
    print("Vérification des fichiers de données...\n")
    
    for file_path, expected_size in REQUIRED_FILES.items():
        if os.path.exists(file_path):
            actual_size = os.path.getsize(file_path)
            size_mb = actual_size / (1024 * 1024)
            existing_files.append(f"✓ {file_path} ({size_mb:.1f}MB)")
        else:
            missing_files.append(f"✗ {file_path} (taille attendue: {expected_size})")
    
    # Afficher les résultats
    if existing_files:
        print("Fichiers présents:")
        for file in existing_files:
            print(f"  {file}")
    
    if missing_files:
        print("\nFichiers manquants:")
        for file in missing_files:
            print(f"  {file}")
        print(f"\n⚠️  {len(missing_files)} fichier(s) manquant(s).")
        print("Consultez DATA_README.md pour savoir comment obtenir ces fichiers.")
        return False
    else:
        print("\n✅ Tous les fichiers requis sont présents!")
        return True

if __name__ == "__main__":
    success = check_files()
    sys.exit(0 if success else 1)