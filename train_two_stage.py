"""
Pipeline complet d'entraînement en 2 étapes :
1. Entraînement du réseau de segmentation du foie
2. Entraînement du réseau de segmentation de la tumeur
"""

import os
import sys

# Ajouter le dossier models au path
sys.path.append('models')

from liver_segmentation import train_liver_segmentation
from tumor_segmentation import train_tumor_segmentation

def main():
    print("🚀 PIPELINE D'ENTRAÎNEMENT EN 2 ÉTAPES")
    print("=" * 50)
    
    # Créer les dossiers nécessaires
    os.makedirs("models", exist_ok=True)
    os.makedirs("weights", exist_ok=True)
    
    # ÉTAPE 1: Entraîner le réseau de segmentation du foie
    print("\n🥇 ÉTAPE 1: Segmentation du FOIE (label 1)")
    liver_model_path = train_liver_segmentation()
    print(f"✅ Modèle foie sauvegardé: {liver_model_path}")
    
    # ÉTAPE 2: Entraîner le réseau de segmentation de la tumeur
    print("\n🥈 ÉTAPE 2: Segmentation de la TUMEUR (label 2)")
    tumor_model_path = train_tumor_segmentation(liver_model_path)
    print(f"✅ Modèle tumeur sauvegardé: {tumor_model_path}")
    
    print("\n🎉 PIPELINE COMPLET TERMINÉ !")
    print(f"📁 Modèles disponibles:")
    print(f"   - Foie: {liver_model_path}")
    print(f"   - Tumeur: {tumor_model_path}")

if __name__ == "__main__":
    main()