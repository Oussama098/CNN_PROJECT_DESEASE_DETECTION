"""
================================================================================
PHASE 3 : ENTRAINEMENT DES MODELES CNN
================================================================================

Ce script effectue l'entrainement complet d'un modele CNN avec :
  - Chargement des donnees pretraitees
  - Construction du modele (transfer learning)
  - Entrainement avec callbacks (early stopping, reduce LR, checkpoint)
  - Sauvegarde de l'historique et des courbes

Usage :
    python 3_training.py --model efficientnetb0
    python 3_training.py --model resnet50 --epochs 30 --lr 0.0005
    python 3_training.py --model densenet121 --batch-size 16

Arguments :
    --model        : Nom du modele a entrainer (defaut: efficientnetb0)
    --epochs       : Nombre maximum d'epochs (defaut: 50)
    --lr           : Taux d'apprentissage (defaut: 0.0001)
    --batch-size   : Taille des batches (defaut: 32)
    --no-augment   : Desactiver la data augmentation
    --fine-tune    : Activer le fine-tuning (defrost couches base)
    --all          : Entrainer tous les modeles

Sortie :
  - checkpoints/{model}_best.keras       : Meilleur modele
  - checkpoints/{model}_epoch_XXX.keras  : Checkpoints par epoch
  - results/{model}_training_log.csv     : Log d'entrainement
  - results/{model}_training_curves.png  : Courbes d'entrainement
  - results/tensorboard/                 : Logs TensorBoard
"""

import os
import sys
import argparse
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from tensorflow import keras

import config
from utils.data_utils import create_data_generators
from utils.model_utils import build_model, get_callbacks
from utils.viz_utils import plot_training_history


def train_model(model_name, epochs=None, batch_size=None, learning_rate=None,
                augment=True, fine_tune=False):
    """
    Entraine un modele CNN specifique.

    Args:
        model_name: Nom du modele ('resnet50', 'densenet121', etc.)
        epochs: Nombre d'epochs (defaut: config.EPOCHS)
        batch_size: Taille des batches (defaut: config.BATCH_SIZE)
        learning_rate: Taux d'apprentissage (defaut: config.LEARNING_RATE)
        augment: Activer la data augmentation
        fine_tune: Defroster les couches de base pour fine-tuning

    Returns:
        model, history
    """
    if epochs is None:
        epochs = config.EPOCHS
    if batch_size is None:
        batch_size = config.BATCH_SIZE
    if learning_rate is None:
        learning_rate = config.LEARNING_RATE

    print("\n" + "=" * 70)
    print(f"  ENTRAINEMENT : {model_name.upper()}")
    print("=" * 70)
    print(f"\n[INFO] Hyperparametres :")
    print(f"  - Epochs                : {epochs}")
    print(f"  - Batch size            : {batch_size}")
    print(f"  - Learning rate         : {learning_rate}")
    print(f"  - Data augmentation     : {'OUI' if augment else 'NON'}")
    print(f"  - Fine-tuning           : {'OUI' if fine_tune else 'NON'}")

    # ============================================================
    # ETAPE 1 : Charger les donnees
    # ============================================================
    print("\n" + "-" * 50)
    print("[1/4] Chargement des donnees")
    print("-" * 50)

    train_gen, valid_gen, class_names, class_indices = create_data_generators(
        augment_train=augment
    )

    print(f"[OK] {len(class_names)} classes chargees")
    print(f"     Train : {train_gen.samples} images ({len(train_gen)} batches)")
    print(f"     Valid : {valid_gen.samples} images ({len(valid_gen)} batches)")

    # ============================================================
    # ETAPE 2 : Construire le modele
    # ============================================================
    print("\n" + "-" * 50)
    print("[2/4] Construction du modele")
    print("-" * 50)

    # Modifier temporairement le LR dans la config
    original_lr = config.LEARNING_RATE
    config.LEARNING_RATE = learning_rate

    model = build_model(
        model_name=model_name,
        num_classes=len(class_names),
        trainable_base=fine_tune,
    )

    config.LEARNING_RATE = original_lr  # Restaurer

    # ============================================================
    # ETAPE 3 : Configurer les callbacks
    # ============================================================
    print("\n" + "-" * 50)
    print("[3/4] Configuration des callbacks")
    print("-" * 50)

    callbacks = get_callbacks(model_name)

    # ============================================================
    # ETAPE 4 : Entrainement
    # ============================================================
    print("\n" + "-" * 50)
    print("[4/4] Entrainement du modele")
    print("-" * 50)

    print(f"\n[INFO] Debut de l'entrainement...")
    print("=" * 70)

    start_time = time.time()

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=valid_gen,
        callbacks=callbacks,
        verbose=1,
    )

    elapsed = time.time() - start_time
    hours = int(elapsed // 3600)
    minutes = int((elapsed % 3600) // 60)
    seconds = int(elapsed % 60)

    print("\n" + "=" * 70)
    print(f"[OK] Entrainement termine en {hours}h {minutes}m {seconds}s")
    print("=" * 70)

    # ============================================================
    # RESUME DES RESULTATS
    # ============================================================
    print("\n" + "-" * 50)
    print("RESULTATS FINaux")
    print("-" * 50)

    best_epoch = history.history["val_accuracy"].index(
        max(history.history["val_accuracy"])
    )

    print(f"\n  Meilleure epoch         : {best_epoch + 1}/{epochs}")
    print(f"  Meilleure val_accuracy  : {max(history.history['val_accuracy']):.4f}")
    print(f"  Meilleure val_loss      : {min(history.history['val_loss']):.4f}")
    print(f"  Val_precision finale    : {history.history['val_precision'][-1]:.4f}")
    print(f"  Val_recall final        : {history.history['val_recall'][-1]:.4f}")

    # ============================================================
    # SAUVEGARDE DES RESULTATS
    # ============================================================
    print("\n" + "-" * 50)
    print("Sauvegarde des resultats")
    print("-" * 50)

    # 1. Courbes d'entrainement
    plot_training_history(history, model_name)

    # 2. Sauvegarder le modele final
    final_model_path = os.path.join(config.MODELS_DIR, f"{model_name}_final.keras")
    model.save(final_model_path)
    print(f"[OK] Modele final sauvegarde : {final_model_path}")

    # 3. Sauvegarder l'historique en JSON
    import json
    history_dict = {
        k: [float(v) for v in vals] for k, vals in history.history.items()
    }
    history_path = os.path.join(config.RESULTS_DIR, f"{model_name}_history.json")
    with open(history_path, "w") as f:
        json.dump(history_dict, f, indent=2)
    print(f"[OK] Historique sauvegarde : {history_path}")

    print("\n" + "=" * 70)
    print(f"  ENTRAINEMENT '{model_name.upper()}' TERMINE")
    print("=" * 70)
    print(f"\n[INFO] Fichiers generes :")
    print(f"       - checkpoints/{model_name}_best.keras")
    print(f"       - models/{model_name}_final.keras")
    print(f"       - results/{model_name}_training_curves.png")
    print(f"       - results/{model_name}_training_log.csv")
    print(f"\n[INFO] Prochaine etape : Executer 'python 4_validation.py --model {model_name}'")
    print("=" * 70 + "\n")

    return model, history


def train_all_models():
    """
    Entraine successivement tous les modeles definis dans config.
    """
    print("\n" + "=" * 70)
    print("  ENTRAINEMENT DE TOUS LES MODELES")
    print(f"  Modeles : {', '.join(config.MODELS_TO_TRAIN)}")
    print("=" * 70)

    results = {}

    for i, model_name in enumerate(config.MODELS_TO_TRAIN, 1):
        print(f"\n\n{'#' * 70}")
        print(f"#  MODELE {i}/{len(config.MODELS_TO_TRAIN)} : {model_name.upper()}")
        print(f"{'#' * 70}")

        try:
            model, history = train_model(model_name)
            best_val_acc = max(history.history["val_accuracy"])
            results[model_name] = {
                "val_accuracy": best_val_acc,
                "val_loss": min(history.history["val_loss"]),
            }
        except Exception as e:
            print(f"\n[ERREUR] Entrainement de '{model_name}' echoue : {e}")
            results[model_name] = {"error": str(e)}

    # Resume comparatif
    print("\n\n" + "=" * 70)
    print("  RESUME COMPARATIF")
    print("=" * 70)
    print(f"\n  {'Modele':<18} {'Val_Accuracy':>12} {'Val_Loss':>12}")
    print("  " + "-" * 45)
    for name, res in results.items():
        if "error" not in res:
            print(f"  {name:<18} {res['val_accuracy']:>12.4f} {res['val_loss']:>12.4f}")
        else:
            print(f"  {name:<18} {'ERREUR':>12} {'ERREUR':>12}")

    return results


def main():
    parser = argparse.ArgumentParser(
        description="Phase 3 : Entrainement des modeles CNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python 3_training.py                          # Modele par defaut
  python 3_training.py --model resnet50         # Entrainer ResNet-50
  python 3_training.py --model densenet121 --epochs 30
  python 3_training.py --model efficientnetb0 --lr 0.0005 --batch-size 16
  python 3_training.py --all                    # Tous les modeles
  python 3_training.py --model mobilenetv2 --fine-tune
        """,
    )
    parser.add_argument(
        "--model",
        default=config.DEFAULT_MODEL,
        choices=["resnet50", "densenet121", "efficientnetb0", "mobilenetv2"],
        help="Modele a entrainer",
    )
    parser.add_argument(
        "--epochs", type=int, default=None, help=f"Nombre d'epochs (defaut: {config.EPOCHS})"
    )
    parser.add_argument(
        "--lr", type=float, default=None, help=f"Learning rate (defaut: {config.LEARNING_RATE})"
    )
    parser.add_argument(
        "--batch-size", type=int, default=None, help=f"Batch size (defaut: {config.BATCH_SIZE})"
    )
    parser.add_argument(
        "--no-augment", action="store_true", help="Desactiver la data augmentation"
    )
    parser.add_argument(
        "--fine-tune", action="store_true", help="Activer le fine-tuning"
    )
    parser.add_argument(
        "--all", action="store_true", help="Entrainer tous les modeles"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  PHASE 3 : ENTRAINEMENT DES MODELES CNN")
    print("=" * 70)

    # Verifier GPU
    gpus = tf.config.list_physical_devices("GPU")
    if gpus:
        print(f"\n[INFO] GPU detecte(s) : {len(gpus)}")
        for gpu in gpus:
            print(f"       - {gpu}")
    else:
        print("\n[AVERTISSEMENT] Aucun GPU detecte. L'entrainement sera lent.")
        print("                Conseil : Utilisez Google Colab ou un environnement GPU.")

    if args.all:
        train_all_models()
    else:
        train_model(
            model_name=args.model,
            epochs=args.epochs,
            batch_size=args.batch_size,
            learning_rate=args.lr,
            augment=not args.no_augment,
            fine_tune=args.fine_tune,
        )

    print("\n" + "=" * 70)
    print("  PHASE 3 TERMINEE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
