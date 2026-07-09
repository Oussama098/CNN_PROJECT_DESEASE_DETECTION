"""
================================================================================
PHASE 5 : ANALYSE ET VISUALISATION AVANCEE
================================================================================

Ce script effectue l'analyse approfondie des resultats :
  - Grad-CAM : Visualisation des regions d'interet du CNN
  - Analyse des erreurs : Classes les plus confondues
  - Comparaison multi-modeles (si plusieurs entraines)
  - Export des resultats pour le rapport final

Usage :
    python 5_analysis.py --model efficientnetb0
    python 5_analysis.py --model resnet50 --gradcam --error-analysis
    python 5_analysis.py --compare-all

Arguments :
    --model          : Nom du modele a analyser
    --gradcam        : Generer les visualisations Grad-CAM
    --error-analysis : Analyser les classes les plus confondues
    --compare-all    : Comparer tous les modeles entraines
    --export         : Exporter les resultats pour le rapport

Sortie :
  - results/{model}_gradcam.png        : Visualisations Grad-CAM
  - results/{model}_error_analysis.png : Analyse des erreurs
  - results/model_comparison.png       : Comparaison des modeles
  - results/final_results.json         : Resultats pour le rapport
"""

import os
import sys
import argparse
import json

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from utils.data_utils import create_data_generators, load_class_mapping
from utils.model_utils import load_trained_model, GRADCAM_LAYERS
from utils.viz_utils import plot_model_comparison

# Style des graphiques
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
sns.set_style("whitegrid")


def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Genere une heatmap Grad-CAM pour une image.

    Args:
        img_array: Image sous forme de tableau numpy
        model: Modele Keras entraine
        last_conv_layer_name: Nom de la derniere couche convolutive
        pred_index: Index de la classe a visualiser (None = classe predite)

    Returns:
        Heatmap Grad-CAM
    """
    # Creer un modele qui mappe l'input a la derniere couche conv + sortie
    grad_model = keras.models.Model(
        model.inputs,
        [model.get_layer(last_conv_layer_name).output, model.output],
    )

    # Calculer le gradient de la classe predite par rapport a la sortie conv
    with tf.GradientTape() as tape:
        conv_outputs, predictions = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(predictions[0])
        class_channel = predictions[:, pred_index]

    # Gradients par rapport a la sortie convolutive
    grads = tape.gradient(class_channel, conv_outputs)

    # Moyenne des gradients spatiaux (pooled grads)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiplier les feature maps par les gradients importants
    conv_outputs = conv_outputs[0]
    heatmap = conv_outputs @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normaliser entre 0 et 1
    heatmap = tf.maximum(heatmap, 0) / (tf.math.reduce_max(heatmap) + 1e-10)

    return heatmap.numpy()


def generate_gradcam(model_name, num_samples=5):
    """
    Genere les visualisations Grad-CAM pour un modele.

    Args:
        model_name: Nom du modele
        num_samples: Nombre d'images a visualiser
    """
    print("\n" + "-" * 50)
    print("[Grad-CAM] Generation des visualisations")
    print("-" * 50)

    # Charger le modele
    try:
        model = load_trained_model(model_name)
    except FileNotFoundError:
        print(f"[ERREUR] Modele '{model_name}' non trouve.")
        return

    # Charger les donnees
    _, valid_gen, class_names, _ = create_data_generators(augment_train=False)
    valid_gen.reset()

    # Obtenir le nom de la couche pour Grad-CAM
    last_conv_layer = GRADCAM_LAYERS.get(model_name)
    if not last_conv_layer:
        print(f"[ERREUR] Couche Grad-CAM non definie pour '{model_name}'")
        return

    print(f"[INFO] Couche Grad-CAM utilisee : {last_conv_layer}")

    # Selectionner des echantillons (corrects et incorrects)
    samples_data = []
    found_correct = 0
    found_incorrect = 0
    max_per_category = num_samples // 2

    for i in range(len(valid_gen)):
        if found_correct >= max_per_category and found_incorrect >= max_per_category:
            break

        x_batch, y_batch = valid_gen[i]
        preds = model.predict(x_batch, verbose=0)

        y_true = np.argmax(y_batch, axis=1)
        y_pred = np.argmax(preds, axis=1)

        for j in range(len(x_batch)):
            is_correct = y_true[j] == y_pred[j]

            if is_correct and found_correct < max_per_category:
                samples_data.append((x_batch[j], y_true[j], y_pred[j], True))
                found_correct += 1
            elif not is_correct and found_incorrect < max_per_category:
                samples_data.append((x_batch[j], y_true[j], y_pred[j], False))
                found_incorrect += 1

        if found_correct >= max_per_category and found_incorrect >= max_per_category:
            break

    if not samples_data:
        print("[AVERTISSEMENT] Aucun echantillon trouve.")
        return

    # Generer les visualisations
    n_samples = len(samples_data)
    fig, axes = plt.subplots(n_samples, 3, figsize=(15, 5 * n_samples))
    if n_samples == 1:
        axes = axes.reshape(1, -1)

    for idx, (img, true_idx, pred_idx, is_correct) in enumerate(samples_data):
        # Preparer l'image
        img_array = np.expand_dims(img, axis=0)

        # Generer la heatmap
        heatmap = make_gradcam_heatmap(img_array, model, last_conv_layer, pred_idx)

        # Superposer heatmap sur l'image
        img_uint8 = (img * 255).astype(np.uint8)
        heatmap_resized = tf.image.resize(
            heatmap[..., np.newaxis], [img.shape[0], img.shape[1]]
        ).numpy()[..., 0]

        # Colormap jet
        heatmap_colored = plt.cm.jet(heatmap_resized)[..., :3]
        superimposed = (img_uint8 / 255.0) * 0.6 + heatmap_colored * 0.4
        superimposed = np.clip(superimposed, 0, 1)

        # Afficher
        # Image originale
        axes[idx, 0].imshow(img_uint8)
        axes[idx, 0].set_title("Image originale", fontsize=10)
        axes[idx, 0].axis("off")

        # Heatmap
        axes[idx, 1].imshow(heatmap_resized, cmap="jet")
        axes[idx, 1].set_title("Grad-CAM Heatmap", fontsize=10)
        axes[idx, 1].axis("off")

        # Superposition
        axes[idx, 2].imshow(superimposed)
        status = "CORRECT" if is_correct else "INCORRECT"
        color = "green" if is_correct else "red"
        axes[idx, 2].set_title(
            f"{status}\nVrai: {class_names[true_idx][:20]}\nPred: {class_names[pred_idx][:20]}",
            fontsize=9,
            color=color,
            fontweight="bold",
        )
        axes[idx, 2].axis("off")

    plt.suptitle(
        f"Grad-CAM - {model_name.upper()}\n"
        "(Regions rouges = zones les plus influentes pour la prediction)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    save_path = os.path.join(config.RESULTS_DIR, f"{model_name}_gradcam.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[OK] Grad-CAM sauvegarde : {save_path}")


def error_analysis(model_name):
    """
    Analyse detaillee des erreurs de classification.

    Args:
        model_name: Nom du modele
    """
    print("\n" + "-" * 50)
    print("[Analyse des erreurs]")
    print("-" * 50)

    try:
        model = load_trained_model(model_name)
    except FileNotFoundError:
        print(f"[ERREUR] Modele '{model_name}' non trouve.")
        return

    _, valid_gen, class_names, _ = create_data_generators(augment_train=False)
    valid_gen.reset()

    # Recolter les predictions
    all_y_true = []
    all_y_pred = []

    for i in range(len(valid_gen)):
        x_batch, y_batch = valid_gen[i]
        preds = model.predict(x_batch, verbose=0)
        all_y_true.extend(np.argmax(y_batch, axis=1))
        all_y_pred.extend(np.argmax(preds, axis=1))

    all_y_true = np.array(all_y_true)
    all_y_pred = np.array(all_y_pred)

    # Trouver les confusions les plus frequentes
    from sklearn.metrics import confusion_matrix
    cm = confusion_matrix(all_y_true, all_y_pred)

    # Masquer la diagonale
    cm_no_diag = cm.copy()
    np.fill_diagonal(cm_no_diag, 0)

    # Trouver les paires les plus confondues
    confusions = []
    for i in range(len(class_names)):
        for j in range(len(class_names)):
            if i != j and cm_no_diag[i, j] > 0:
                confusions.append((
                    class_names[i],
                    class_names[j],
                    cm_no_diag[i, j],
                    cm_no_diag[i, j] / cm[i].sum() if cm[i].sum() > 0 else 0,
                ))

    confusions.sort(key=lambda x: x[2], reverse=True)

    # Afficher les top confusions
    print(f"\n[INFO] Top 10 des confusions les plus frequentes :")
    print(f"{'':4s} {'Classe reelle':<30s} {'-> Classe predite':<30s} {'Count':>6s} {'%':>6s}")
    print("  " + "-" * 80)
    for i, (true_cls, pred_cls, count, pct) in enumerate(confusions[:10], 1):
        print(f"  {i:2d} {true_cls:<30s} -> {pred_cls:<30s} {count:>5d} {pct:>6.1%}")

    # Visualisation des erreurs par classe
    error_rates = []
    class_labels = []
    for i, name in enumerate(class_names):
        total = cm[i].sum()
        if total > 0:
            errors = total - cm[i, i]
            error_rates.append(errors / total)
            class_labels.append(name[:25])

    # Trier par taux d'erreur decroissant
    sorted_data = sorted(zip(error_rates, class_labels), reverse=True)
    error_rates, class_labels = zip(*sorted_data)

    fig, ax = plt.subplots(figsize=(12, max(6, len(class_labels) * 0.35)))
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(error_rates)))
    bars = ax.barh(range(len(class_labels)), error_rates, color=colors)
    ax.set_yticks(range(len(class_labels)))
    ax.set_yticklabels(class_labels, fontsize=8)
    ax.set_xlabel("Taux d'erreur", fontsize=10)
    ax.set_title(
        f"Taux d'erreur par classe - {model_name.upper()}",
        fontsize=12,
        fontweight="bold",
    )
    ax.set_xlim(0, 1)
    ax.grid(axis="x", alpha=0.3)

    # Ajouter les valeurs
    for bar, val in zip(bars, error_rates):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.2%}", va="center", fontsize=8)

    plt.tight_layout()
    save_path = os.path.join(config.RESULTS_DIR, f"{model_name}_error_analysis.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"\n[OK] Analyse des erreurs sauvegardee : {save_path}")

    return confusions


def compare_all_models():
    """
    Compare tous les modeles entraines et genere un rapport comparatif.
    """
    print("\n" + "-" * 50)
    print("[Comparaison de tous les modeles]")
    print("-" * 50)

    results_dict = {}

    for model_name in config.MODELS_TO_TRAIN:
        # Charger le rapport de classification
        report_path = os.path.join(
            config.RESULTS_DIR, f"{model_name}_classification_report.json"
        )
        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                report = json.load(f)

            results_dict[model_name] = {
                "accuracy": report.get("accuracy", 0),
                "precision": report.get("weighted avg", {}).get("precision", 0),
                "recall": report.get("weighted avg", {}).get("recall", 0),
                "f1": report.get("weighted avg", {}).get("f1-score", 0),
            }
            print(f"  [OK] {model_name}: accuracy={results_dict[model_name]['accuracy']:.4f}")
        else:
            print(f"  [SKIP] {model_name}: rapport non trouve")

    if len(results_dict) > 1:
        plot_model_comparison(results_dict)
    else:
        print("[INFO] Pas assez de modeles pour une comparaison.")

    return results_dict


def export_results():
    """
    Exporte tous les resultats dans un fichier JSON pour le rapport final.
    """
    print("\n" + "-" * 50)
    print("[Export des resultats]")
    print("-" * 50)

    final_results = {
        "project": "Detection des maladies des plantes par CNN",
        "dataset": "New Plant Diseases Dataset (87K+ images, 38 classes)",
        "models": {},
        "references": {
            "Mohanty et al. (2016)": {"model": "GoogLeNet+TL", "accuracy": 0.9935},
            "Ferentinos (2018)": {"model": "VGG+TL", "accuracy": 0.9953},
            "Brahimi et al. (2018)": {"model": "InceptionV3+TL", "accuracy": 0.9976},
        },
    }

    for model_name in config.MODELS_TO_TRAIN:
        report_path = os.path.join(
            config.RESULTS_DIR, f"{model_name}_classification_report.json"
        )
        history_path = os.path.join(
            config.RESULTS_DIR, f"{model_name}_history.json"
        )

        model_data = {}

        if os.path.exists(report_path):
            with open(report_path, "r") as f:
                report = json.load(f)
            model_data["classification_report"] = report

        if os.path.exists(history_path):
            with open(history_path, "r") as f:
                history = json.load(f)
            model_data["training_history"] = {
                k: v[-1] if v else 0 for k, v in history.items()
            }
            model_data["best_val_accuracy"] = max(history.get("val_accuracy", [0]))
            model_data["epochs_trained"] = len(history.get("accuracy", []))

        if model_data:
            final_results["models"][model_name] = model_data

    export_path = os.path.join(config.RESULTS_DIR, "final_results.json")
    with open(export_path, "w", encoding="utf-8") as f:
        json.dump(final_results, f, indent=2, ensure_ascii=False)

    print(f"[OK] Resultats exportes : {export_path}")
    return final_results


def main():
    parser = argparse.ArgumentParser(
        description="Phase 5 : Analyse et visualisation avancee",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python 5_analysis.py --model efficientnetb0 --gradcam
  python 5_analysis.py --model resnet50 --error-analysis
  python 5_analysis.py --model densenet121 --gradcam --error-analysis --export
  python 5_analysis.py --compare-all
  python 5_analysis.py --export
        """,
    )
    parser.add_argument(
        "--model",
        default=config.DEFAULT_MODEL,
        choices=["resnet50", "densenet121", "efficientnetb0", "mobilenetv2"],
        help="Modele a analyser",
    )
    parser.add_argument(
        "--gradcam", action="store_true", help="Generer les visualisations Grad-CAM"
    )
    parser.add_argument(
        "--error-analysis", action="store_true", help="Analyser les erreurs"
    )
    parser.add_argument(
        "--compare-all", action="store_true", help="Comparer tous les modeles"
    )
    parser.add_argument(
        "--export", action="store_true", help="Exporter les resultats"
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  PHASE 5 : ANALYSE ET VISUALISATION AVANCEE")
    print("=" * 70)

    # Si aucun flag specifique, tout activer
    if not any([args.gradcam, args.error_analysis, args.compare_all, args.export]):
        args.gradcam = True
        args.error_analysis = True
        args.export = True

    if args.gradcam:
        generate_gradcam(args.model, num_samples=config.NUM_GRADCAM_SAMPLES * 2)

    if args.error_analysis:
        error_analysis(args.model)

    if args.compare_all:
        compare_all_models()

    if args.export:
        export_results()

    print("\n" + "=" * 70)
    print("  PHASE 5 TERMINEE")
    print("=" * 70)
    print(f"\n[INFO] Toutes les analyses sont sauvegardees dans : {config.RESULTS_DIR}/")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
