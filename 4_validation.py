"""
================================================================================
PHASE 4 : VALIDATION ET EVALUATION DES MODELES
================================================================================

Ce script evalue les performances d'un modele entraine sur l'ensemble de
validation en calculant toutes les metriques requises :
  - Accuracy globale
  - Precision, Recall, F1-score (par classe et moyennes)
  - Top-3 Accuracy
  - Matrice de confusion
  - Rapport de classification detaille

Usage :
    python 4_validation.py --model efficientnetb0
    python 4_validation.py --model resnet50 --checkpoint best
    python 4_validation.py --all

Arguments :
    --model       : Nom du modele a evaluer
    --checkpoint  : Type de checkpoint ('best' ou 'last')
    --all         : Evaluer tous les modeles disponibles

Sortie :
  - results/{model}_confusion_matrix.png      : Matrice de confusion
  - results/{model}_classification_report.txt : Rapport texte
  - results/{model}_classification_report.json: Rapport JSON
  - results/{model}_sample_predictions.png    : Exemples de predictions
"""

import os
import sys
import argparse
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import tensorflow as tf
from tensorflow import keras

import config
from utils.data_utils import create_data_generators, load_class_mapping
from utils.model_utils import load_trained_model
from utils.viz_utils import (
    plot_confusion_matrix,
    plot_sample_predictions,
    save_classification_report,
    plot_class_performance,
)


def evaluate_model(model_name, checkpoint_type="best"):
    """
    Evalue un modele sur l'ensemble de validation.

    Args:
        model_name: Nom du modele
        checkpoint_type: 'best' ou 'last'

    Returns:
        Dictionnaire des resultats d'evaluation
    """
    print("\n" + "=" * 70)
    print(f"  VALIDATION : {model_name.upper()}")
    print("=" * 70)

    # ============================================================
    # ETAPE 1 : Charger le modele
    # ============================================================
    print("\n" + "-" * 50)
    print("[1/4] Chargement du modele")
    print("-" * 50)

    try:
        model = load_trained_model(model_name, checkpoint_type)
    except FileNotFoundError:
        print(f"\n[ERREUR] Modele non trouve pour '{model_name}'")
        print(f"         Executez d'abord : python 3_training.py --model {model_name}")
        return None

    # ============================================================
    # ETAPE 2 : Charger les donnees de validation
    # ============================================================
    print("\n" + "-" * 50)
    print("[2/4] Chargement des donnees de validation")
    print("-" * 50)

    # Pas d'augmentation pour la validation
    _, valid_gen, class_names, _ = create_data_generators(augment_train=False)

    print(f"[OK] {valid_gen.samples} images de validation chargees")

    # ============================================================
    # ETAPE 3 : Evaluation globale
    # ============================================================
    print("\n" + "-" * 50)
    print("[3/4] Evaluation globale")
    print("-" * 50)

    results = model.evaluate(valid_gen, verbose=1, return_dict=True)

    print("\n" + "=" * 50)
    print("  METRIQUES GLOBALES")
    print("=" * 50)
    for metric_name, value in results.items():
        print(f"  {metric_name:<25s} : {value:.4f}")

    # ============================================================
    # ETAPE 4 : Predictions detaillees
    # ============================================================
    print("\n" + "-" * 50)
    print("[4/4] Predictions et metriques par classe")
    print("-" * 50)

    # Reinitialiser le generateur
    valid_gen.reset()

    # Recolter toutes les predictions (mais garder seulement un echantillon d'images)
    all_y_true = []
    all_y_pred = []
    sample_images = []
    sample_y_true = []
    sample_y_pred = []
    SAMPLE_LIMIT = 32  # largement suffisant pour plot_sample_predictions (16 utilisees)

    # Calculer le nombre de batches
    n_batches = len(valid_gen)

    for i in range(n_batches):
        x_batch, y_batch = valid_gen[i]
        preds = model.predict(x_batch, verbose=0)

        y_true_batch = np.argmax(y_batch, axis=1)
        y_pred_batch = np.argmax(preds, axis=1)

        all_y_true.extend(y_true_batch)
        all_y_pred.extend(y_pred_batch)

        if len(sample_images) < SAMPLE_LIMIT:
            remaining = SAMPLE_LIMIT - len(sample_images)
            sample_images.extend(x_batch[:remaining])
            sample_y_true.extend(y_true_batch[:remaining])
            sample_y_pred.extend(y_pred_batch[:remaining])

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)
    sample_images = np.array(sample_images)
    sample_y_true = np.array(sample_y_true)
    sample_y_pred = np.array(sample_y_pred)
    # ============================================================
    # RAPPORT DE CLASSIFICATION
    # ============================================================
    print("\n[INFO] Generation du rapport de classification...")
    report = save_classification_report(
        all_y_true, all_y_pred, class_names, model_name
    )

    # Afficher les moyennes
    print("\n" + "=" * 50)
    print("  MOYENNES")
    print("=" * 50)
    for avg_type in ["macro avg", "weighted avg"]:
        if avg_type in report:
            p = report[avg_type]["precision"]
            r = report[avg_type]["recall"]
            f1 = report[avg_type]["f1-score"]
            print(f"  {avg_type:<15s} Precision: {p:.4f}  Recall: {r:.4f}  F1: {f1:.4f}")

    # ============================================================
    # VISUALISATIONS
    # ============================================================
    print("\n[INFO] Generation des visualisations...")

    # 1. Matrice de confusion (complete)
    plot_confusion_matrix(all_y_true, all_y_pred, class_names, model_name)

    # 2. Matrice de confusion (top 15 classes les plus confondues)
    plot_confusion_matrix(
        all_y_true, all_y_pred, class_names, model_name, top_n=15
    )

    plot_sample_predictions(
        sample_images, sample_y_true, sample_y_pred, class_names, model_name, num_samples=16
    )

    # 4. Performances par classe
    plot_class_performance(report, class_names, model_name)

    # ============================================================
    # RESUME
    # ============================================================
    print("\n" + "=" * 70)
    print(f"  VALIDATION '{model_name.upper()}' TERMINEE")
    print("=" * 70)
    print(f"\n[INFO] Fichiers generes :")
    print(f"       - results/{model_name}_confusion_matrix.png")
    print(f"       - results/{model_name}_confusion_matrix_top15.png")
    print(f"       - results/{model_name}_sample_predictions.png")
    print(f"       - results/{model_name}_class_performance.png")
    print(f"       - results/{model_name}_classification_report.txt")
    print(f"       - results/{model_name}_classification_report.json")
    print(f"\n[INFO] Prochaine etape : Executer 'python 5_analysis.py --model {model_name}'")
    print("=" * 70 + "\n")

    return {
        "model": model_name,
        "accuracy": results.get("accuracy", 0),
        "precision": report.get("weighted avg", {}).get("precision", 0),
        "recall": report.get("weighted avg", {}).get("recall", 0),
        "f1": report.get("weighted avg", {}).get("f1-score", 0),
        "top3_accuracy": results.get("top3_accuracy", 0),
    }


def evaluate_all_models():
    """
    Evalue tous les modeles disponibles et genere une comparaison.
    """
    print("\n" + "=" * 70)
    print("  VALIDATION DE TOUS LES MODELES")
    print("=" * 70)

    all_results = {}

    for model_name in config.MODELS_TO_TRAIN:
        try:
            res = evaluate_model(model_name)
            if res:
                all_results[model_name] = res
        except Exception as e:
            print(f"\n[ERREUR] Evaluation de '{model_name}' echouee : {e}")

    # Comparaison
    if len(all_results) > 1:
        from utils.viz_utils import plot_model_comparison
        plot_model_comparison(all_results)

    return all_results


def main():
    parser = argparse.ArgumentParser(
        description="Phase 4 : Validation et evaluation des modeles",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python 4_validation.py --model efficientnetb0
  python 4_validation.py --model resnet50 --checkpoint best
  python 4_validation.py --all
        """,
    )
    parser.add_argument(
        "--model",
        default=config.DEFAULT_MODEL,
        choices=["resnet50", "densenet121", "efficientnetb0", "mobilenetv2"],
        help="Modele a evaluer",
    )
    parser.add_argument(
        "--checkpoint",
        default="best",
        choices=["best", "last"],
        help="Type de checkpoint a charger",
    )
    parser.add_argument(
        "--all", action="store_true", help="Evaluer tous les modeles"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  PHASE 4 : VALIDATION ET EVALUATION")
    print("=" * 70)

    if args.all:
        evaluate_all_models()
    else:
        evaluate_model(args.model, args.checkpoint)

    print("\n" + "=" * 70)
    print("  PHASE 4 TERMINEE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
