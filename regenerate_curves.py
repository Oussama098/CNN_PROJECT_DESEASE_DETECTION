"""
regenerate_curves.py
Regenere les courbes d'entrainement (accuracy/loss/precision/recall) a partir
des fichiers results/{model}_training_log.csv, meme si l'entrainement a ete
interrompu avant la fin (session expiree, etc.)

Usage:
    python regenerate_curves.py --model efficientnetb0
    python regenerate_curves.py --all
"""

import os
import sys
import argparse
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config


def load_history_from_csv(model_name):
    """
    Charge l'historique d'entrainement depuis le fichier CSV genere par CSVLogger.
    Retourne un dict compatible avec plot_training_history / history.json.
    """
    csv_path = os.path.join(config.RESULTS_DIR, f"{model_name}_training_log.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"Fichier introuvable : {csv_path}")

    history = {}
    with open(csv_path, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            for key, value in row.items():
                if key == "epoch":
                    continue
                history.setdefault(key, []).append(float(value))

    return history


def plot_curves_from_history(history, model_name):
    """
    Genere les 4 sous-graphiques (Accuracy, Loss, Precision, Recall)
    a partir d'un dict d'historique, meme partiel.
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle(f"Courbes d'entrainement - {model_name.upper()}", fontsize=14, fontweight="bold")

    pairs = [
        ("accuracy", "val_accuracy", "Accuracy", axes[0, 0]),
        ("loss", "val_loss", "Loss", axes[0, 1]),
        ("precision", "val_precision", "Precision", axes[1, 0]),
        ("recall", "val_recall", "Recall", axes[1, 1]),
    ]

    epochs = range(1, len(history.get("accuracy", [])) + 1)

    for train_key, val_key, title, ax in pairs:
        if train_key in history:
            ax.plot(epochs, history[train_key], label="Train", color="blue", linewidth=2)
        if val_key in history:
            ax.plot(epochs, history[val_key], label="Validation", color="red", linewidth=2)
        ax.set_title(title, fontsize=12, fontweight="bold")
        ax.set_xlabel("Epoch")
        ax.set_ylabel(title)
        ax.legend()
        ax.grid(alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(config.RESULTS_DIR, f"{model_name}_training_curves.png")
    plt.savefig(save_path, bbox_inches="tight", dpi=150)
    plt.close()
    print(f"[OK] Courbes regenerees : {save_path}")


def save_history_json(history, model_name):
    """
    Sauvegarde l'historique au format JSON (comme le fait 3_training.py normalement),
    pour que 5_analysis.py --compare-all et --export puissent le retrouver.
    """
    history_path = os.path.join(config.RESULTS_DIR, f"{model_name}_history.json")
    with open(history_path, "w") as f:
        json.dump(history, f, indent=2)
    print(f"[OK] history.json regenere : {history_path}")


def regenerate(model_name):
    print(f"\n[INFO] Regeneration pour '{model_name}'...")
    try:
        history = load_history_from_csv(model_name)
    except FileNotFoundError as e:
        print(f"[SKIP] {e}")
        return

    n_epochs = len(history.get("accuracy", []))
    print(f"[INFO] {n_epochs} epochs trouvees dans le CSV")

    if "val_accuracy" in history:
        best_acc = max(history["val_accuracy"])
        best_epoch = history["val_accuracy"].index(best_acc) + 1
        print(f"[INFO] Meilleure val_accuracy : {best_acc:.4f} (epoch {best_epoch})")

    plot_curves_from_history(history, model_name)
    save_history_json(history, model_name)


def main():
    parser = argparse.ArgumentParser(description="Regenere les courbes depuis les CSV de log")
    parser.add_argument(
        "--model", default=None,
        choices=["resnet50", "densenet121", "efficientnetb0", "mobilenetv2"],
        help="Modele a regenerer"
    )
    parser.add_argument("--all", action="store_true", help="Regenerer tous les modeles")
    args = parser.parse_args()

    if args.all:
        for model_name in config.MODELS_TO_TRAIN:
            regenerate(model_name)
    elif args.model:
        regenerate(args.model)
    else:
        print("[ERREUR] Precisez --model NOM ou --all")


if __name__ == "__main__":
    main()