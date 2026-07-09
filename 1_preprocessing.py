"""
================================================================================
PHASE 1 : PRETRAITEMENT DES DONNEES
================================================================================

Ce script effectue le pretraitement complet du dataset :
  - Chargement des images depuis le dossier du dataset
  - Data augmentation (rotation, zoom, flip, luminosite)
  - Normalisation des pixels
  - Creation des generateurs train/validation
  - Affichage des statistiques du dataset
  - Sauvegarde du mapping des classes

Usage :
    python 1_preprocessing.py

Sortie :
  - results/class_mapping.json  : Correspondance indices <-> classes
  - Statistiques affichees dans la console
"""

import os
import sys

# Ajouter le repertoire du projet au path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from utils.data_utils import (
    get_dataset_info,
    create_data_generators,
    get_class_names,
    get_dataset_paths,
)


def main():
    print("=" * 70)
    print("  PHASE 1 : PRETRAITEMENT DES DONNEES")
    print("  Projet : Detection des maladies des plantes par CNN")
    print("=" * 70)

    # ============================================================
    # ETAPE 1 : Verifier le dataset
    # ============================================================
    print("\n" + "-" * 50)
    print("[1/4] Verification du dataset")
    print("-" * 50)

    try:
        train_dir, valid_dir = get_dataset_paths()
        print(f"[OK] Dossier train trouve : {train_dir}")
        print(f"[OK] Dossier valid trouve : {valid_dir}")
    except FileNotFoundError as e:
        print(f"\n[ERREUR] {e}")
        print("\n" + "=" * 50)
        print("INSTRUCTIONS POUR TELECHARGER LE DATASET :")
        print("=" * 50)
        print("""
1. Allez sur https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset
2. Telechargez le dataset (1.43 GB)
3. Extrayez l'archive dans le dossier 'data/' du projet
4. La structure attendue est :
   data/New Plant Diseases Dataset(Augmented)/
   └── New Plant Diseases Dataset(Augmented)/
       ├── train/  (38 sous-dossiers = 38 classes)
       └── valid/  (38 sous-dossiers = 38 classes)
        """)
        sys.exit(1)

    # ============================================================
    # ETAPE 2 : Statistiques du dataset
    # ============================================================
    print("\n" + "-" * 50)
    print("[2/4] Analyse des statistiques")
    print("-" * 50)

    info = get_dataset_info()

    # ============================================================
    # ETAPE 3 : Creation des generateurs de donnees
    # ============================================================
    print("\n" + "-" * 50)
    print("[3/4] Creation des generateurs de donnees")
    print("-" * 50)

    print("\n[INFO] Parametres de pretraitement :")
    print(f"  - Taille des images     : {config.IMG_SIZE}x{config.IMG_SIZE}")
    print(f"  - Taille des batches    : {config.BATCH_SIZE}")
    print(f"  - Data augmentation     : OUI")
    print(f"    * Rotation            : +/-{config.ROTATION_RANGE} degres")
    print(f"    * Zoom                : {config.ZOOM_RANGE}")
    print(f"    * Flip horizontal     : {config.HORIZONTAL_FLIP}")
    print(f"    * Flip vertical       : {config.VERTICAL_FLIP}")
    print(f"    * Luminosite          : {config.BRIGHTNESS_RANGE}")
    print(f"    * Cisaillement        : {config.SHEAR_RANGE}")
    print(f"  - Normalisation         : Rescaling 1/255")
    print(f"  - Graine aleatoire      : {config.RANDOM_SEED}")

    train_gen, valid_gen, class_names, class_indices = create_data_generators(
        augment_train=True
    )

    print(f"\n[OK] Generateurs crees avec succes")
    print(f"     - Batches d'entrainement   : {len(train_gen)}")
    print(f"     - Batches de validation    : {len(valid_gen)}")
    print(f"     - Nombre de classes        : {len(class_names)}")

    # ============================================================
    # ETAPE 4 : Verification des batches
    # ============================================================
    print("\n" + "-" * 50)
    print("[4/4] Verification des batches")
    print("-" * 50)

    # Verifier un batch d'entrainement
    x_batch, y_batch = next(train_gen)
    print(f"\n[INFO] Batch d'entrainement :")
    print(f"  - Shape des images  : {x_batch.shape}")
    print(f"  - Shape des labels  : {y_batch.shape}")
    print(f"  - Plage des pixels  : [{x_batch.min():.3f}, {x_batch.max():.3f}]")
    print(f"  - Type des donnees  : {x_batch.dtype}")

    # Verifier un batch de validation
    x_val, y_val = next(valid_gen)
    print(f"\n[INFO] Batch de validation :")
    print(f"  - Shape des images  : {x_val.shape}")
    print(f"  - Shape des labels  : {y_val.shape}")
    print(f"  - Plage des pixels  : [{x_val.min():.3f}, {x_val.max():.3f}]")

    # ============================================================
    # RESUME
    # ============================================================
    print("\n" + "=" * 70)
    print("  PHASE 1 TERMINEE AVEC SUCCES")
    print("=" * 70)
    print(f"\n[INFO] Fichiers generes :")
    mapping_path = os.path.join(config.RESULTS_DIR, "class_mapping.json")
    print(f"       - {mapping_path}")
    print(f"\n[INFO] Prochaine etape : Executer 'python 2_cnn_models.py'")
    print("=" * 70 + "\n")

    return train_gen, valid_gen, class_names, class_indices


if __name__ == "__main__":
    main()
