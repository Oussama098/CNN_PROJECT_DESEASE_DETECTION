"""
================================================================================
PHASE 2 : DEVELOPPEMENT CNN - ARCHITECTURES
================================================================================

Ce script definit et construit les architectures CNN pour la classification
des maladies des plantes. Quatre architectures modernes sont implementees :

  1. ResNet-50      (~25M parametres) - Connexions residuelles
  2. DenseNet-121   (~8M parametres)  - Connexions denses
  3. EfficientNet-B0 (~5.3M params)   - Meilleur rapport precision/efficacite
  4. MobileNetV2    (~3.5M parametres) - Architecture legere (mobile)

Toutes utilisent le TRANSFER LEARNING depuis ImageNet, conformement aux
conclusions de l'etat de l'art (Mohanty et al., Ferentinos, Brahimi et al.).

Usage :
    python 2_cnn_models.py [nom_modele]

Arguments :
    nom_modele : 'resnet50', 'densenet121', 'efficientnetb0', 'mobilenetv2'
                 (par defaut : tous les modeles sont construits)

Sortie :
  - Resume de chaque architecture dans la console
  - Modeles sauvegardes au format .keras dans models/
"""

import os
import sys
import argparse

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from utils.model_utils import build_model, PREPROCESSING_FNS, GRADCAM_LAYERS


def display_model_summary(model, model_name):
    """
    Affiche un resume formate du modele.
    """
    print("\n" + "=" * 60)
    print(f"  RESUME DU MODELE : {model_name.upper()}")
    print("=" * 60)

    # Architecture
    model.summary()

    # Statistiques
    total_params = model.count_params()
    trainable_params = sum(
        w.numpy().size for w in model.trainable_weights
    )
    non_trainable_params = total_params - trainable_params

    print("\n" + "-" * 40)
    print(f"  Parametres totaux         : {total_params:>12,}")
    print(f"  Parametres entrainables   : {trainable_params:>12,}")
    print(f"  Parametres non-entr.      : {non_trainable_params:>12,}")
    print(f"  Couche Grad-CAM           : {GRADCAM_LAYERS.get(model_name, 'N/A')}")
    print("-" * 40)


def save_model_architecture(model, model_name):
    """
    Sauvegarde l'architecture du modele (sans poids).
    """
    # Sauvegarder le modele complet
    model_path = os.path.join(config.MODELS_DIR, f"{model_name}_architecture.keras")
    model.save(model_path)
    print(f"[INFO] Architecture sauvegardee : {model_path}")

    # Sauvegarder un resume texte
    summary_path = os.path.join(config.MODELS_DIR, f"{model_name}_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        model.summary(print_fn=lambda x: f.write(x + "\n"))
    print(f"[INFO] Resume texte sauvegarde : {summary_path}")


def build_all_models():
    """
    Construit tous les modeles definis dans config.MODELS_TO_TRAIN.
    """
    print("\n" + "=" * 70)
    print("  CONSTRUCTION DE TOUS LES MODELES")
    print(f"  Modeles a construire : {', '.join(config.MODELS_TO_TRAIN)}")
    print("=" * 70)

    built_models = {}

    for model_name in config.MODELS_TO_TRAIN:
        try:
            model = build_model(model_name)
            display_model_summary(model, model_name)
            save_model_architecture(model, model_name)
            built_models[model_name] = model
            print(f"\n[OK] Modele '{model_name}' construit avec succes\n")
        except Exception as e:
            print(f"\n[ERREUR] Construction de '{model_name}' echouee : {e}\n")

    # Resume comparatif
    print("\n" + "=" * 70)
    print("  COMPARAISON DES ARCHITECTURES")
    print("=" * 70)
    print(f"\n  {'Modele':<18} {'Parametres':>12} {'Entrainables':>12} {'Couche Grad-CAM'}")
    print("  " + "-" * 65)
    for name, model in built_models.items():
        total = model.count_params()
        trainable = sum(w.numpy().size for w in model.trainable_weights)
        gradcam = GRADCAM_LAYERS.get(name, "N/A")
        print(f"  {name:<18} {total:>12,} {trainable:>12,} {gradcam}")

    print("\n" + "=" * 70)
    print("  REFERENCES DE L'ETAT DE L'ART (PlantVillage)")
    print("=" * 70)
    print("  Mohanty et al. (2016)  : GoogLeNet + TL  = 99.35%")
    print("  Ferentinos (2018)      : VGG + TL        = 99.53%")
    print("  Brahimi et al. (2018)  : InceptionV3 + TL= 99.76%")
    print("  Brahimi et al. (2018)  : DenseNet-169+TL = 99.72%")
    print("=" * 70)

    return built_models


def build_single_model(model_name):
    """
    Construit un seul modele specifie.
    """
    if model_name not in PREPROCESSING_FNS:
        print(f"[ERREUR] Modele '{model_name}' non reconnu.")
        print(f"         Choix possibles : {list(PREPROCESSING_FNS.keys())}")
        sys.exit(1)

    print("=" * 70)
    print(f"  CONSTRUCTION DU MODELE : {model_name.upper()}")
    print("=" * 70)

    model = build_model(model_name)
    display_model_summary(model, model_name)
    save_model_architecture(model, model_name)

    print(f"\n[OK] Modele '{model_name}' pret pour l'entrainement")
    print(f"[INFO] Prochaine etape : Executer 'python 3_training.py --model {model_name}'")

    return {model_name: model}


def main():
    parser = argparse.ArgumentParser(
        description="Phase 2 : Developpement des architectures CNN",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples :
  python 2_cnn_models.py                  # Construire tous les modeles
  python 2_cnn_models.py resnet50         # Construire uniquement ResNet-50
  python 2_cnn_models.py efficientnetb0   # Construire uniquement EfficientNet-B0
        """,
    )
    parser.add_argument(
        "model",
        nargs="?",
        default="all",
        choices=["all", "resnet50", "densenet121", "efficientnetb0", "mobilenetv2"],
        help="Modele a construire (defaut: all)",
    )
    args = parser.parse_args()

    print("=" * 70)
    print("  PHASE 2 : DEVELOPPEMENT CNN - ARCHITECTURES")
    print("=" * 70)

    if args.model == "all":
        build_all_models()
    else:
        build_single_model(args.model)

    print("\n" + "=" * 70)
    print("  PHASE 2 TERMINEE")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
