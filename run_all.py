"""
================================================================================
SCRIPT MAITRE - Execution sequentielle de toutes les phases
================================================================================

Ce script execute automatiquement les 5 phases du projet en sequence :
  1. Pretraitement des donnees
  2. Developpement des architectures CNN
  3. Entrainement des modeles
  4. Validation et evaluation
  5. Analyse avancee

Usage :
    python run_all.py [--model efficientnetb0] [--epochs 50]

Arguments :
    --model     : Modele a entrainer (defaut: efficientnetb0)
    --epochs    : Nombre d'epochs (defaut: 50)
    --skip      : Phases a ignorer (ex: "2,5")
"""

import os
import sys
import argparse
import subprocess


def run_phase(script_name, description, extra_args=None):
    """
    Execute un script de phase et verifie le code de retour.
    """
    print("\n" + "=" * 70)
    print(f"  {description}")
    print("=" * 70)

    cmd = [sys.executable, script_name]
    if extra_args:
        cmd.extend(extra_args)

    result = subprocess.run(cmd, cwd=os.path.dirname(os.path.abspath(__file__)))

    if result.returncode != 0:
        print(f"\n[ERREUR] La phase '{script_name}' a echoue (code {result.returncode})")
        response = input("Voulez-vous continuer ? [O/n] : ").strip().lower()
        if response == 'n':
            sys.exit(1)

    return result.returncode == 0


def main():
    parser = argparse.ArgumentParser(
        description="Execution sequentielle de toutes les phases"
    )
    parser.add_argument(
        "--model",
        default="efficientnetb0",
        choices=["resnet50", "densenet121", "efficientnetb0", "mobilenetv2"],
        help="Modele principal a entrainer",
    )
    parser.add_argument(
        "--epochs", type=int, default=50, help="Nombre d'epochs"
    )
    parser.add_argument(
        "--skip",
        default="",
        help="Phases a ignorer (ex: '2,5' pour ignorer phases 2 et 5)",
    )
    parser.add_argument(
        "--gpu", action="store_true", help="Forcer l'utilisation du GPU"
    )
    args = parser.parse_args()

    skip_phases = [p.strip() for p in args.skip.split(",") if p.strip()]

    print("=" * 70)
    print("  PIPELINE COMPLET - DETECTION DES MALADIES DES PLANTES")
    print("=" * 70)
    print(f"\n[INFO] Modele selectionne : {args.model.upper()}")
    print(f"[INFO] Epochs             : {args.epochs}")
    print(f"[INFO] Phases ignorees    : {', '.join(skip_phases) if skip_phases else 'Aucune'}")

    # Configurer le GPU si demande
    if args.gpu:
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    phases = [
        ("1_preprocessing.py", "PHASE 1 : PRETRAITEMENT", []),
        ("2_cnn_models.py", "PHASE 2 : DEVELOPPEMENT CNN", [args.model]),
        ("3_training.py", "PHASE 3 : ENTRAINEMENT",
         ["--model", args.model, "--epochs", str(args.epochs)]),
        ("4_validation.py", "PHASE 4 : VALIDATION",
         ["--model", args.model]),
        ("5_analysis.py", "PHASE 5 : ANALYSE",
         ["--model", args.model, "--gradcam", "--error-analysis", "--export"]),
    ]

    for i, (script, desc, extra) in enumerate(phases, 1):
        phase_num = str(i)
        if phase_num in skip_phases:
            print(f"\n[INFO] Phase {i} ignoree.")
            continue
        run_phase(script, desc, extra)

    print("\n" + "=" * 70)
    print("  PIPELINE TERMINE AVEC SUCCES")
    print("=" * 70)
    print("\n[INFO] Les resultats sont disponibles dans le dossier 'results/'")
    print("[INFO] Les modeles sont sauvegardes dans 'checkpoints/' et 'models/'")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()
