"""
utils/data_utils.py
Utilitaires pour le chargement et le pretraitement des donnees.
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import sys
import json

# Ajouter le repertoire parent au path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
import config


def get_dataset_paths():
    """
    Retourne les chemins vers les dossiers train et valid.
    Verifie que le dataset existe.
    """
    train_dir = os.path.join(config.DATASET_PATH, "train")
    valid_dir = os.path.join(config.DATASET_PATH, "valid")

    if not os.path.exists(train_dir):
        raise FileNotFoundError(
            f"Dossier train non trouve : {train_dir}\n"
            "Verifiez que le dataset est bien telecharge depuis Kaggle :\n"
            "https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset"
        )
    if not os.path.exists(valid_dir):
        raise FileNotFoundError(
            f"Dossier valid non trouve : {valid_dir}\n"
            "Verifiez la structure du dataset."
        )

    return train_dir, valid_dir


def get_class_names(train_dir):
    """
    Extrait et retourne la liste des noms de classes.
    """
    class_names = sorted([
        d for d in os.listdir(train_dir)
        if os.path.isdir(os.path.join(train_dir, d))
    ])
    return class_names


def create_data_generators(augment_train=True):
    """
    Cree les generateurs de donnees pour l'entrainement et la validation.

    Args:
        augment_train: Si True, applique la data augmentation sur le train set.

    Returns:
        train_generator, valid_generator, class_names, class_indices
    """
    train_dir, valid_dir = get_dataset_paths()
    class_names = get_class_names(train_dir)

    print(f"[INFO] Classes trouvees : {len(class_names)}")
    print(f"[INFO] Premieres classes : {class_names[:5]}...")

    # --- Generateur d'entrainement ---
    if augment_train:
        train_datagen = ImageDataGenerator(
            rotation_range=config.ROTATION_RANGE,
            zoom_range=config.ZOOM_RANGE,
            horizontal_flip=config.HORIZONTAL_FLIP,
            vertical_flip=config.VERTICAL_FLIP,
            brightness_range=config.BRIGHTNESS_RANGE,
            shear_range=config.SHEAR_RANGE,
            fill_mode="nearest",
        )
        print("[INFO] Data augmentation activee pour l'entrainement")
    else:
        train_datagen = ImageDataGenerator()
        print("[INFO] Pas de data augmentation (normalisation uniquement)")

    # --- Generateur de validation (pas d'augmentation) ---
    valid_datagen = ImageDataGenerator()

    # --- Chargement des donnees ---
    train_generator = train_datagen.flow_from_directory(
        train_dir,
        target_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
        class_mode="categorical",
        shuffle=True,
        seed=config.RANDOM_SEED,
    )

    valid_generator = valid_datagen.flow_from_directory(
        valid_dir,
        target_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
        class_mode="categorical",
        shuffle=False,  # Important pour la confusion matrix
        seed=config.RANDOM_SEED,
    )

    # Sauvegarder les indices de classes
    class_indices = train_generator.class_indices
    save_class_mapping(class_indices, class_names)

    return train_generator, valid_generator, class_names, class_indices


def save_class_mapping(class_indices, class_names):
    """
    Sauvegarde la correspondance entre indices et noms de classes.
    """
    # Inverser le dictionnaire : index -> nom
    idx_to_class = {v: k for k, v in class_indices.items()}

    mapping_path = os.path.join(config.RESULTS_DIR, "class_mapping.json")
    with open(mapping_path, "w", encoding="utf-8") as f:
        json.dump({
            "class_indices": class_indices,
            "idx_to_class": idx_to_class,
            "class_names": class_names,
            "num_classes": len(class_names),
        }, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Mapping des classes sauvegarde : {mapping_path}")


def load_class_mapping():
    """
    Charge le mapping des classes depuis le fichier JSON.
    """
    mapping_path = os.path.join(config.RESULTS_DIR, "class_mapping.json")
    with open(mapping_path, "r", encoding="utf-8") as f:
        return json.load(f)


def get_dataset_info():
    """
    Affiche les statistiques du dataset.
    """
    train_dir, valid_dir = get_dataset_paths()
    class_names = get_class_names(train_dir)

    # Compter les images
    train_counts = {}
    valid_counts = {}

    for cls in class_names:
        train_cls_dir = os.path.join(train_dir, cls)
        valid_cls_dir = os.path.join(valid_dir, cls)

        train_counts[cls] = len([
            f for f in os.listdir(train_cls_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])
        valid_counts[cls] = len([
            f for f in os.listdir(valid_cls_dir)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])

    total_train = sum(train_counts.values())
    total_valid = sum(valid_counts.values())

    print("\n" + "=" * 60)
    print("STATISTIQUES DU DATASET")
    print("=" * 60)
    print(f"Nombre de classes        : {len(class_names)}")
    print(f"Images d'entrainement    : {total_train}")
    print(f"Images de validation     : {total_valid}")
    print(f"Total                    : {total_train + total_valid}")
    print(f"\nRepartition par classe (Top 10) :")

    sorted_classes = sorted(train_counts.items(), key=lambda x: x[1], reverse=True)
    for cls, count in sorted_classes[:10]:
        valid_count = valid_counts.get(cls, 0)
        print(f"  {cls:40s} : {count:5d} train / {valid_count:5d} valid")

    if len(sorted_classes) > 10:
        print(f"  ... et {len(sorted_classes) - 10} autres classes")

    print("=" * 60 + "\n")

    return {
        "num_classes": len(class_names),
        "total_train": total_train,
        "total_valid": total_valid,
        "class_names": class_names,
        "train_counts": train_counts,
        "valid_counts": valid_counts,
    }


def create_tf_dataset(preprocess_fn=None):
    """
    Cree des datasets TensorFlow optimises (alternatif aux ImageDataGenerator).
    Utilise tf.data pour de meilleures performances.

    Args:
        preprocess_fn: Fonction de pretraitement (ex: keras.applications.resnet50.preprocess_input)

    Returns:
        train_ds, val_ds, class_names
    """
    train_dir, valid_dir = get_dataset_paths()

    train_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        image_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
        label_mode="categorical",
        shuffle=True,
        seed=config.RANDOM_SEED,
    )

    val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        valid_dir,
        image_size=(config.IMG_SIZE, config.IMG_SIZE),
        batch_size=config.BATCH_SIZE,
        label_mode="categorical",
        shuffle=False,
        seed=config.RANDOM_SEED,
    )

    class_names = train_ds.class_names

    # Appliquer la normalisation et le preprocessing du modele
    normalization_layer = tf.keras.layers.Rescaling(1.0 / 255.0)

    if preprocess_fn:
        def preprocess(x, y):
            x = normalization_layer(x)
            x = preprocess_fn(x)
            return x, y
    else:
        def preprocess(x, y):
            x = normalization_layer(x)
            return x, y

    train_ds = train_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)
    val_ds = val_ds.map(preprocess, num_parallel_calls=tf.data.AUTOTUNE)

    # Optimisation des performances
    train_ds = train_ds.prefetch(tf.data.AUTOTUNE)
    val_ds = val_ds.prefetch(tf.data.AUTOTUNE)

    # Sauvegarder le mapping
    class_indices = {name: i for i, name in enumerate(class_names)}
    save_class_mapping(class_indices, class_names)

    return train_ds, val_ds, class_names


if __name__ == "__main__":
    # Test des fonctions
    print("Test de data_utils.py")
    info = get_dataset_info()
    print(f"\n[INFO] Test OK - {info['num_classes']} classes chargees")
