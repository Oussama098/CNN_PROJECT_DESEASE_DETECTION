"""
evaluate_plantdoc.py
Evalue un modele entraine sur PlantVillage contre le dataset PlantDoc
(images en conditions reelles / terrain), avec mapping automatique des
classes par similarite de texte (les noms de classes different entre
les deux datasets).

Usage:
    python evaluate_plantdoc.py --model densenet121 --plantdoc-dir /kaggle/input/plantdoc-dataset
"""

import os
import sys
import json
import argparse
from difflib import SequenceMatcher

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, classification_report

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from utils.model_utils import load_trained_model
from utils.data_utils import load_class_mapping
from tensorflow.keras.preprocessing.image import load_img, img_to_array


def normalize_text(s):
    """Nettoie un nom de classe pour comparaison (minuscules, sans separateurs)."""
    return s.lower().replace("_", " ").replace("-", " ").replace("___", " ").strip()


def similarity(a, b):
    """Score de similarite [0,1] entre deux chaines."""
    return SequenceMatcher(None, normalize_text(a), normalize_text(b)).ratio()


def build_class_mapping(plantdoc_classes, plantvillage_classes, threshold=0.35):
    """
    Associe chaque classe PlantDoc a la classe PlantVillage la plus proche
    par similarite textuelle. Retourne un dict {plantdoc_class: plantvillage_class or None}.
    """
    mapping = {}
    print("\n" + "=" * 70)
    print("  MAPPING AUTOMATIQUE DES CLASSES (PlantDoc -> PlantVillage)")
    print("=" * 70)

    for pd_class in plantdoc_classes:
        best_match = None
        best_score = 0.0
        for pv_class in plantvillage_classes:
            score = similarity(pd_class, pv_class)
            if score > best_score:
                best_score = score
                best_match = pv_class

        if best_score >= threshold:
            mapping[pd_class] = best_match
            status = "OK"
        else:
            mapping[pd_class] = None
            status = "SKIP (pas de correspondance fiable)"

        print(f"  {pd_class:45s} -> {str(best_match):45s} (score={best_score:.2f}) [{status}]")

    print("=" * 70)
    matched = sum(1 for v in mapping.values() if v is not None)
    print(f"\n[INFO] {matched}/{len(plantdoc_classes)} classes PlantDoc mappees avec succes")
    print("[ATTENTION] Verifiez ce mapping ci-dessus - corrigez manuellement si besoin")
    print("            en editant le dict retourne avant de lancer l'evaluation.\n")

    return mapping


def load_plantdoc_images(plantdoc_dir, class_mapping, img_size=224, max_per_class=None):
    """
    Charge les images de PlantDoc dont la classe a ete mappee avec succes.
    Retourne (images, true_labels_plantvillage_names, plantdoc_class_names).
    """
    images = []
    true_labels = []
    source_classes = []

    for pd_class, pv_class in class_mapping.items():
        if pv_class is None:
            continue

        class_dir = os.path.join(plantdoc_dir, pd_class)
        if not os.path.isdir(class_dir):
            continue

        files = [f for f in os.listdir(class_dir) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        if max_per_class:
            files = files[:max_per_class]

        for fname in files:
            fpath = os.path.join(class_dir, fname)
            try:
                img = load_img(fpath, target_size=(img_size, img_size))
                img_array = img_to_array(img)  # [0,255], pas de rescale (Lambda interne au modele)
                images.append(img_array)
                true_labels.append(pv_class)
                source_classes.append(pd_class)
            except Exception as e:
                print(f"[SKIP] {fpath}: {e}")

    return np.array(images), true_labels, source_classes


def evaluate_on_plantdoc(model_name, plantdoc_dir, max_per_class=None, manual_mapping_path=None):
    print("\n" + "=" * 70)
    print(f"  EVALUATION CROSS-DATASET : {model_name.upper()} sur PlantDoc")
    print("=" * 70)

    # 1. Charger le modele et le mapping de classes PlantVillage
    model = load_trained_model(model_name, checkpoint_type="best")
    pv_mapping = load_class_mapping()
    pv_class_names = pv_mapping["class_names"]
    class_indices = pv_mapping["class_indices"]  # {class_name: idx}

    # 2. Lister les classes PlantDoc disponibles
    plantdoc_classes = sorted([
        d for d in os.listdir(plantdoc_dir)
        if os.path.isdir(os.path.join(plantdoc_dir, d))
    ])
    print(f"[INFO] {len(plantdoc_classes)} classes trouvees dans PlantDoc")

    # 3. Construire le mapping automatique (a verifier/corriger si besoin)
    class_mapping = build_class_mapping(plantdoc_classes, pv_class_names)

    # 3bis. Appliquer les corrections manuelles si fournies (ecrase le mapping auto)
    if manual_mapping_path and os.path.exists(manual_mapping_path):
        with open(manual_mapping_path, "r", encoding="utf-8") as f:
            manual_overrides = json.load(f)
        print(f"\n[INFO] Application de {len(manual_overrides)} corrections manuelles depuis {manual_mapping_path}")
        for pd_class, pv_class in manual_overrides.items():
            old_value = class_mapping.get(pd_class, "N/A")
            class_mapping[pd_class] = pv_class
            print(f"  [CORRIGE] {pd_class:35s} : {old_value} -> {pv_class}")

    # 4. Charger les images correspondantes
    print("[INFO] Chargement des images PlantDoc...")
    images, true_labels, source_classes = load_plantdoc_images(
        plantdoc_dir, class_mapping, img_size=config.IMG_SIZE, max_per_class=max_per_class
    )
    print(f"[OK] {len(images)} images chargees et mappees")

    if len(images) == 0:
        print("[ERREUR] Aucune image mappee - verifiez le mapping ci-dessus")
        return

    # 5. Predictions
    print("[INFO] Predictions en cours...")
    predictions = model.predict(images, verbose=1, batch_size=32)
    pred_indices = np.argmax(predictions, axis=1)
    idx_to_class = {v: k for k, v in class_indices.items()}
    pred_labels = [idx_to_class[i] for i in pred_indices]

    # 6. Metriques
    true_indices = [class_indices[c] for c in true_labels]
    accuracy = np.mean(np.array(pred_indices) == np.array(true_indices))

    print("\n" + "=" * 50)
    print("  RESULTATS CROSS-DATASET (PlantDoc)")
    print("=" * 50)
    print(f"  Accuracy globale : {accuracy:.4f} ({accuracy*100:.2f}%)")
    print(f"  Images evaluees  : {len(images)}")

    # Comparaison avec la performance sur PlantVillage (a renseigner manuellement)
    print(f"\n  Rappel - accuracy sur PlantVillage (validation) : voir final_results.json")
    print("=" * 50)

    # 7. Rapport de classification (uniquement sur les classes presentes dans le sous-ensemble)
    unique_classes = sorted(set(true_labels))
    report = classification_report(
        true_labels, pred_labels, labels=unique_classes, zero_division=0, output_dict=True
    )
    report_path = os.path.join(config.RESULTS_DIR, f"{model_name}_plantdoc_report.json")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    print(f"[OK] Rapport sauvegarde : {report_path}")

    # 8. Graphique comparatif par classe (accuracy PlantDoc par classe)
    class_accuracies = {}
    for cls in unique_classes:
        idx = [i for i, t in enumerate(true_labels) if t == cls]
        cls_acc = np.mean(np.array(pred_indices)[idx] == class_indices[cls])
        class_accuracies[cls] = cls_acc

    plt.figure(figsize=(10, max(6, len(class_accuracies) * 0.3)))
    classes_sorted = sorted(class_accuracies.items(), key=lambda x: x[1])
    plt.barh([c[0] for c in classes_sorted], [c[1] for c in classes_sorted], color="steelblue")
    plt.xlabel("Accuracy sur PlantDoc")
    plt.title(f"{model_name.upper()} - Performance par classe sur PlantDoc (conditions reelles)")
    plt.xlim(0, 1)
    plt.tight_layout()
    plot_path = os.path.join(config.RESULTS_DIR, f"{model_name}_plantdoc_class_accuracy.png")
    plt.savefig(plot_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"[OK] Graphique sauvegarde : {plot_path}")

    # 9. Sauvegarde du mapping utilise (pour tracabilite/reproductibilite)
    mapping_path = os.path.join(config.RESULTS_DIR, f"{model_name}_plantdoc_class_mapping_used.json")
    with open(mapping_path, "w") as f:
        json.dump(class_mapping, f, indent=2)
    print(f"[OK] Mapping utilise sauvegarde : {mapping_path}")

    return {
        "model": model_name,
        "accuracy": float(accuracy),
        "n_images": len(images),
        "n_classes_mapped": len(unique_classes),
    }


def main():
    parser = argparse.ArgumentParser(description="Evaluation cross-dataset sur PlantDoc")
    parser.add_argument("--model", required=True,
                         choices=["resnet50", "densenet121", "efficientnetb0", "mobilenetv2"])
    parser.add_argument("--plantdoc-dir", required=True, help="Chemin vers le dossier PlantDoc (train ou test)")
    parser.add_argument("--max-per-class", type=int, default=None,
                         help="Limiter le nombre d'images par classe (utile pour un test rapide)")
    parser.add_argument("--manual-mapping", type=str, default=None,
                         help="Chemin vers un JSON de corrections manuelles {plantdoc_class: plantvillage_class} "
                              "qui ecrase le mapping automatique pour les classes mal appariees")
    args = parser.parse_args()

    result = evaluate_on_plantdoc(args.model, args.plantdoc_dir, args.max_per_class, args.manual_mapping)
    print("\n" + "=" * 70)
    print("  EVALUATION TERMINEE")
    print("=" * 70)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
