"""
utils/viz_utils.py
Utilitaires de visualisation pour l'analyse des resultats.
"""

import os
import sys
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Backend non-interactif
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import config

# Style des graphiques
plt.rcParams["figure.dpi"] = 150
plt.rcParams["savefig.dpi"] = 150
plt.rcParams["figure.figsize"] = (10, 6)
sns.set_style("whitegrid")


def plot_training_history(history, model_name, save_dir=None):
    """
    Trace les courbes d'entrainement (accuracy et loss).

    Args:
        history: Historique d'entrainement (history.history ou CSV)
        model_name: Nom du modele
        save_dir: Repertoire de sauvegarde
    """
    if save_dir is None:
        save_dir = config.RESULTS_DIR

    os.makedirs(save_dir, exist_ok=True)

    # Extraire les donnees
    if hasattr(history, "history"):
        hist = history.history
    else:
        hist = history

    epochs = range(1, len(hist["accuracy"]) + 1)

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Accuracy
    ax = axes[0, 0]
    ax.plot(epochs, hist["accuracy"], "b-", label="Train", linewidth=2)
    ax.plot(epochs, hist["val_accuracy"], "r-", label="Validation", linewidth=2)
    ax.set_title("Accuracy", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Loss
    ax = axes[0, 1]
    ax.plot(epochs, hist["loss"], "b-", label="Train", linewidth=2)
    ax.plot(epochs, hist["val_loss"], "r-", label="Validation", linewidth=2)
    ax.set_title("Loss", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Loss")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Precision
    ax = axes[1, 0]
    ax.plot(epochs, hist["precision"], "b-", label="Train", linewidth=2)
    ax.plot(epochs, hist["val_precision"], "r-", label="Validation", linewidth=2)
    ax.set_title("Precision", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Precision")
    ax.legend()
    ax.grid(True, alpha=0.3)

    # Recall
    ax = axes[1, 1]
    ax.plot(epochs, hist["recall"], "b-", label="Train", linewidth=2)
    ax.plot(epochs, hist["val_recall"], "r-", label="Validation", linewidth=2)
    ax.set_title("Recall", fontsize=12, fontweight="bold")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Recall")
    ax.legend()
    ax.grid(True, alpha=0.3)

    fig.suptitle(
        f"Courbes d'entrainement - {model_name.upper()}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{model_name}_training_curves.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Courbes sauvegardees : {save_path}")


def plot_confusion_matrix(y_true, y_pred, class_names, model_name, save_dir=None, top_n=None):
    """
    Trace et sauvegarde la matrice de confusion.

    Args:
        y_true: Labels reels (indices)
        y_pred: Labels predits (indices)
        class_names: Liste des noms de classes
        model_name: Nom du modele
        save_dir: Repertoire de sauvegarde
        top_n: Si specifie, n'affiche que les top_n classes les plus confondues
    """
    if save_dir is None:
        save_dir = config.RESULTS_DIR

    os.makedirs(save_dir, exist_ok=True)

    # Calculer la matrice de confusion
    cm = confusion_matrix(y_true, y_pred)

    # Si top_n est specifie, selectionner les classes les plus confondues
    if top_n and top_n < len(class_names):
        # Trouver les classes avec le plus d'erreurs (hors diagonale)
        cm_copy = cm.copy()
        np.fill_diagonal(cm_copy, 0)
        error_counts = cm_copy.sum(axis=1) + cm_copy.sum(axis=0)
        top_indices = np.argsort(error_counts)[-top_n:]
        cm = cm[np.ix_(top_indices, top_indices)]
        class_names = [class_names[i] for i in top_indices]

    # Normaliser par ligne (pourcentages)
    cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
    cm_norm = np.nan_to_num(cm_norm)

    fig, axes = plt.subplots(1, 2, figsize=(20, 9))

    # Matrice brute
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[0],
        cbar_kws={"label": "Nombre d'images"},
    )
    axes[0].set_title("Matrice de confusion (brute)", fontweight="bold")
    axes[0].set_xlabel("Classe predite")
    axes[0].set_ylabel("Classe reelle")
    plt.setp(axes[0].get_xticklabels(), rotation=45, ha="right", fontsize=7)
    plt.setp(axes[0].get_yticklabels(), rotation=0, fontsize=7)

    # Matrice normalisee
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="YlOrRd",
        xticklabels=class_names,
        yticklabels=class_names,
        ax=axes[1],
        vmin=0,
        vmax=1,
        cbar_kws={"label": "Proportion"},
    )
    axes[1].set_title("Matrice de confusion (normalisee)", fontweight="bold")
    axes[1].set_xlabel("Classe predite")
    axes[1].set_ylabel("Classe reelle")
    plt.setp(axes[1].get_xticklabels(), rotation=45, ha="right", fontsize=7)
    plt.setp(axes[1].get_yticklabels(), rotation=0, fontsize=7)

    fig.suptitle(
        f"Matrice de confusion - {model_name.upper()}",
        fontsize=14,
        fontweight="bold",
    )
    plt.tight_layout()

    suffix = f"_top{top_n}" if top_n else ""
    save_path = os.path.join(save_dir, f"{model_name}_confusion_matrix{suffix}.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Matrice de confusion sauvegardee : {save_path}")


def plot_sample_predictions(images, y_true, y_pred, class_names, model_name, save_dir=None, num_samples=16):
    """
    Affiche un echantillon d'images avec leurs predictions.

    Args:
        images: Batch d'images
        y_true: Labels reels (indices)
        y_pred: Labels predits (indices)
        class_names: Liste des noms de classes
        model_name: Nom du modele
        save_dir: Repertoire de sauvegarde
        num_samples: Nombre d'images a afficher
    """
    if save_dir is None:
        save_dir = config.RESULTS_DIR

    os.makedirs(save_dir, exist_ok=True)

    num_samples = min(num_samples, len(images))
    cols = 4
    rows = (num_samples + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(16, 4 * rows))
    axes = axes.flatten() if rows > 1 else (axes.flatten() if hasattr(axes, "flatten") else [axes])

    for i in range(num_samples):
        ax = axes[i]
        img = images[i]

        # Denormaliser si necessaire
        if img.max() <= 1.0:
            img = img * 255.0
        img = img.astype(np.uint8)

        true_label = class_names[y_true[i]]
        pred_label = class_names[y_pred[i]]
        is_correct = y_true[i] == y_pred[i]

        ax.imshow(img)
        color = "green" if is_correct else "red"
        title = f"Vrai: {true_label[:20]}\nPred: {pred_label[:20]}"
        ax.set_title(title, color=color, fontsize=8, fontweight="bold")
        ax.axis("off")

    # Cacher les axes vides
    for i in range(num_samples, len(axes)):
        axes[i].axis("off")

    fig.suptitle(
        f"Predictions - {model_name.upper()} (vert=correct, roux=incorrect)",
        fontsize=12,
        fontweight="bold",
    )
    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{model_name}_sample_predictions.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Echantillon de predictions sauvegarde : {save_path}")


def plot_class_performance(report_dict, class_names, model_name, save_dir=None):
    """
    Visualise les performances par classe (precision, recall, F1).

    Args:
        report_dict: Dictionnaire du classification_report
        class_names: Liste des noms de classes
        model_name: Nom du modele
        save_dir: Repertoire de sauvegarde
    """
    if save_dir is None:
        save_dir = config.RESULTS_DIR

    os.makedirs(save_dir, exist_ok=True)

    # Extraire les metriques par classe
    classes = [c for c in report_dict.keys() if c not in ["accuracy", "macro avg", "weighted avg"]]

    precisions = [report_dict[c]["precision"] for c in classes]
    recalls = [report_dict[c]["recall"] for c in classes]
    f1_scores = [report_dict[c]["f1-score"] for c in classes]

    # Mapping des noms
    display_names = [class_names[int(c)] if c.isdigit() else c for c in classes]

    # Trier par F1-score decroissant
    sorted_indices = np.argsort(f1_scores)[::-1]
    display_names = [display_names[i] for i in sorted_indices]
    precisions = [precisions[i] for i in sorted_indices]
    recalls = [recalls[i] for i in sorted_indices]
    f1_scores = [f1_scores[i] for i in sorted_indices]

    fig, ax = plt.subplots(figsize=(14, max(8, len(display_names) * 0.3)))

    y_pos = np.arange(len(display_names))
    bar_height = 0.25

    ax.barh(y_pos + bar_height, precisions, bar_height, label="Precision", color="#3498db")
    ax.barh(y_pos, recalls, bar_height, label="Recall", color="#2ecc71")
    ax.barh(y_pos - bar_height, f1_scores, bar_height, label="F1-Score", color="#e74c3c")

    ax.set_yticks(y_pos)
    ax.set_yticklabels([n[:30] for n in display_names], fontsize=8)
    ax.set_xlabel("Score", fontsize=10)
    ax.set_title(
        f"Performances par classe - {model_name.upper()}",
        fontsize=12,
        fontweight="bold",
    )
    ax.legend(loc="lower right")
    ax.set_xlim(0, 1.05)
    ax.grid(axis="x", alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(save_dir, f"{model_name}_class_performance.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Performances par classe sauvegardees : {save_path}")


def plot_model_comparison(results_dict, save_dir=None):
    """
    Compare les performances de plusieurs modeles.

    Args:
        results_dict: Dictionnaire {model_name: {accuracy, precision, recall, f1}}
        save_dir: Repertoire de sauvegarde
    """
    if save_dir is None:
        save_dir = config.RESULTS_DIR

    os.makedirs(save_dir, exist_ok=True)

    models = list(results_dict.keys())
    metrics = ["accuracy", "precision", "recall", "f1"]
    metric_labels = ["Accuracy", "Precision", "Recall", "F1-Score"]

    fig, ax = plt.subplots(figsize=(12, 7))

    x = np.arange(len(models))
    width = 0.18
    colors = ["#3498db", "#2ecc71", "#f39c12", "#e74c3c"]

    for i, (metric, label, color) in enumerate(zip(metrics, metric_labels, colors)):
        values = [results_dict[m].get(metric, 0) for m in models]
        offset = (i - 1.5) * width
        bars = ax.bar(x + offset, values, width, label=label, color=color, edgecolor="white")

        # Ajouter les valeurs sur les barres
        for bar, val in zip(bars, values):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8, fontweight="bold")

    ax.set_ylabel("Score", fontsize=11)
    ax.set_title(
        "Comparaison des architectures CNN",
        fontsize=13,
        fontweight="bold",
    )
    ax.set_xticks(x)
    ax.set_xticklabels([m.upper() for m in models], fontsize=10)
    ax.legend(loc="lower right")
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", alpha=0.3)

    plt.tight_layout()

    save_path = os.path.join(save_dir, "model_comparison.png")
    plt.savefig(save_path, bbox_inches="tight")
    plt.close()
    print(f"[INFO] Comparaison des modeles sauvegardee : {save_path}")


def save_classification_report(y_true, y_pred, class_names, model_name, save_dir=None):
    """
    Genere et sauvegarde le rapport de classification au format texte.

    Args:
        y_true: Labels reels
        y_pred: Labels predits
        class_names: Liste des noms de classes
        model_name: Nom du modele
        save_dir: Repertoire de sauvegarde

    Returns:
        Dictionnaire du rapport de classification
    """
    if save_dir is None:
        save_dir = config.RESULTS_DIR

    os.makedirs(save_dir, exist_ok=True)

    # Generer le rapport
    report = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
        output_dict=True,
    )

    # Version texte
    report_text = classification_report(
        y_true, y_pred,
        target_names=class_names,
        digits=4,
    )

    # Sauvegarder en texte
    report_path = os.path.join(save_dir, f"{model_name}_classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(f"RAPPORT DE CLASSIFICATION - {model_name.upper()}\n")
        f.write("=" * 60 + "\n\n")
        f.write(report_text)

    # Sauvegarder en JSON
    report_json_path = os.path.join(save_dir, f"{model_name}_classification_report.json")
    with open(report_json_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2, ensure_ascii=False)

    print(f"[INFO] Rapport de classification sauvegarde :")
    print(f"       - Texte : {report_path}")
    print(f"       - JSON  : {report_json_path}")

    return report


if __name__ == "__main__":
    print("Test de viz_utils.py - Module de visualisation")
