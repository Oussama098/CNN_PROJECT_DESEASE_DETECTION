"""
utils/model_utils.py
Utilitaires pour la construction des modeles CNN.
"""

import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import (
    ResNet50,
    DenseNet121,
    EfficientNetB0,
    MobileNetV2,
)


# ============================================================
# DICTIONNAIRE DES FONCTIONS DE PREPROCESSING
# ============================================================
PREPROCESSING_FNS = {
    "resnet50": tf.keras.applications.resnet50.preprocess_input,
    "densenet121": tf.keras.applications.densenet.preprocess_input,
    "efficientnetb0": tf.keras.applications.efficientnet.preprocess_input,
    "mobilenetv2": tf.keras.applications.mobilenet_v2.preprocess_input,
}

# ============================================================
# DICTIONNAIRE DES NOMS DE COUCHES POUR GRAD-CAM
# ============================================================
GRADCAM_LAYERS = {
    "resnet50": "conv5_block3_out",
    "densenet121": "conv5_block16_concat",
    "efficientnetb0": "top_conv",
    "mobilenetv2": "Conv_1",
}


def build_model(model_name, num_classes=None, img_size=None, trainable_base=False):
    """
    Construit un modele CNN avec transfer learning.

    Args:
        model_name: Nom du modele ('resnet50', 'densenet121', 'efficientnetb0', 'mobilenetv2')
        num_classes: Nombre de classes (defaut: config.NUM_CLASSES)
        img_size: Taille des images (defaut: config.IMG_SIZE)
        trainable_base: Si True, entraine aussi les couches de base.

    Returns:
        Model: Le modele Keras compile
    """
    if num_classes is None:
        num_classes = config.NUM_CLASSES
    if img_size is None:
        img_size = config.IMG_SIZE

    input_shape = (img_size, img_size, 3)
    weights = "imagenet"

    print(f"\n[INFO] Construction du modele : {model_name.upper()}")
    print(f"[INFO] Input shape : {input_shape}")
    print(f"[INFO] Classes : {num_classes}")
    print(f"[INFO] Poids pre-entraines : {weights}")
    print(f"[INFO] Base trainable : {trainable_base}")

    # --- Charger le modele de base (backbone) ---
    if model_name == "resnet50":
        base_model = ResNet50(
            weights=weights,
            include_top=False,
            input_shape=input_shape,
        )
    elif model_name == "densenet121":
        base_model = DenseNet121(
            weights=weights,
            include_top=False,
            input_shape=input_shape,
        )
    elif model_name == "efficientnetb0":
        base_model = EfficientNetB0(
            weights=weights,
            include_top=False,
            input_shape=input_shape,
        )
    elif model_name == "mobilenetv2":
        base_model = MobileNetV2(
            weights=weights,
            include_top=False,
            input_shape=input_shape,
        )
    else:
        raise ValueError(
            f"Modele '{model_name}' non supporte. "
            f"Choix possibles : {list(PREPROCESSING_FNS.keys())}"
        )

    # Geler ou defroster les couches de base
    base_model.trainable = trainable_base
    if not trainable_base:
        print("[INFO] Couches de base gelees (transfer learning)")
    else:
        print("[INFO] Couches de base defrostees (fine-tuning)")

    # --- Construire la tete de classification ---
    inputs = keras.Input(shape=input_shape, name="input_layer")

    # Preprocessing specifique au modele (attend des pixels [0,255])
    preprocess_fn = PREPROCESSING_FNS[model_name]
    x = layers.Lambda(preprocess_fn, name="model_preprocessing")(inputs)

    # Passer par le backbone
    x = base_model(x, training=trainable_base)

    # Couches de classification (tete personnalisee)
    x = layers.GlobalAveragePooling2D(name="global_avg_pool")(x)
    x = layers.BatchNormalization(name="bn_head")(x)
    x = layers.Dropout(0.5, name="dropout_1")(x)
    x = layers.Dense(512, activation="relu", name="dense_1")(x)
    x = layers.BatchNormalization(name="bn_dense")(x)
    x = layers.Dropout(0.3, name="dropout_2")(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions",
        dtype="float32",  # Pour la stabilite numerique en mixed precision
    )(x)

    # Creer le modele final
    model = Model(inputs, outputs, name=f"plant_disease_{model_name}")

    # --- Compiler le modele ---
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=[
            "accuracy",
            keras.metrics.Precision(name="precision"),
            keras.metrics.Recall(name="recall"),
            keras.metrics.TopKCategoricalAccuracy(k=3, name="top3_accuracy"),
        ],
    )

    print(f"[INFO] Modele compile avec succes")
    print(f"[INFO] Parametres totaux : {model.count_params():,}")
    print(f"[INFO] Parametres entrainables : {sum(tf.keras.backend.count_params(w) for w in model.trainable_weights):,}")

    return model


def get_callbacks(model_name):
    """
    Cree les callbacks pour l'entrainement.

    Args:
        model_name: Nom du modele (pour les noms de fichiers)

    Returns:
        Liste de callbacks Keras
    """
    callbacks = []

    # 1. Early Stopping
    early_stopping = keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=config.EARLY_STOPPING_PATIENCE,
        restore_best_weights=True,
        verbose=1,
        mode="max",
    )
    callbacks.append(early_stopping)

    # 2. Reduction du learning rate
    reduce_lr = keras.callbacks.ReduceLROnPlateau(
        monitor="val_loss",
        factor=config.REDUCE_LR_FACTOR,
        patience=config.REDUCE_LR_PATIENCE,
        min_lr=config.MIN_LR,
        verbose=1,
        mode="min",
    )
    callbacks.append(reduce_lr)

    # 3. Sauvegarde du meilleur modele
    checkpoint_path = os.path.join(config.CHECKPOINT_DIR, f"{model_name}_best.keras")
    checkpoint = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_path,
        monitor="val_accuracy",
        save_best_only=True,
        verbose=1,
        mode="max",
    )
    callbacks.append(checkpoint)

    # 4. Sauvegarde a chaque epoch (pour reprendre si besoin)
    checkpoint_all_path = os.path.join(
        config.CHECKPOINT_DIR, f"{model_name}_epoch_{{epoch:03d}}.keras"
    )
    checkpoint_all = keras.callbacks.ModelCheckpoint(
        filepath=checkpoint_all_path,
        save_freq="epoch",
        verbose=0,
    )
    callbacks.append(checkpoint_all)

    # 5. Logging CSV
    csv_path = os.path.join(config.RESULTS_DIR, f"{model_name}_training_log.csv")
    csv_logger = keras.callbacks.CSVLogger(csv_path, append=True)
    callbacks.append(csv_logger)

    # 6. TensorBoard (optionnel)
    log_dir = os.path.join(config.RESULTS_DIR, "tensorboard", model_name)
    tensorboard = keras.callbacks.TensorBoard(
        log_dir=log_dir,
        histogram_freq=1,
        update_freq="epoch",
    )
    callbacks.append(tensorboard)

    print(f"[INFO] Callbacks configures :")
    print(f"       - Early stopping (patience={config.EARLY_STOPPING_PATIENCE})")
    print(f"       - Reduce LR (patience={config.REDUCE_LR_PATIENCE})")
    print(f"       - Model checkpoint : {checkpoint_path}")
    print(f"       - CSV logger : {csv_path}")
    print(f"       - TensorBoard : {log_dir}")

    return callbacks


def load_trained_model(model_name, checkpoint_type="best"):
    """
    Charge un modele entraine depuis un checkpoint.

    Args:
        model_name: Nom du modele
        checkpoint_type: 'best' ou 'last'

    Returns:
        Modele Keras charge
    """
    if checkpoint_type == "best":
        checkpoint_path = os.path.join(
            config.CHECKPOINT_DIR, f"{model_name}_best.keras"
        )
    else:
        # Trouver le dernier checkpoint
        checkpoints = [
            f for f in os.listdir(config.CHECKPOINT_DIR)
            if f.startswith(f"{model_name}_epoch_") and f.endswith(".keras")
        ]
        if not checkpoints:
            raise FileNotFoundError(f"Aucun checkpoint trouve pour {model_name}")
        checkpoints.sort()
        checkpoint_path = os.path.join(config.CHECKPOINT_DIR, checkpoints[-1])

    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint non trouve : {checkpoint_path}")

    custom_objects = {"preprocess_input": PREPROCESSING_FNS[model_name]}
    model = keras.models.load_model(
        checkpoint_path, custom_objects=custom_objects, safe_mode=False
    )
    print(f"[INFO] Modele charge avec succes")

    return model


def enable_mixed_precision():
    """
    Active la mixed precision pour accelerer l'entrainement sur GPU compatibles.
    """
    policy = tf.keras.mixed_precision.Policy("mixed_float16")
    tf.keras.mixed_precision.set_global_policy(policy)
    print("[INFO] Mixed precision activee (float16)")


if __name__ == "__main__":
    # Test
    print("Test de model_utils.py")
    for name in ["resnet50", "densenet121", "efficientnetb0", "mobilenetv2"]:
        model = build_model(name)
        print(f"  {name}: {model.count_params():,} parametres\n")
