"""
config.py
Configuration centrale pour le projet de detection de maladies des plantes.
Tous les parametres modifiables sont centralises ici.
"""

import os

# ============================================================
# CHEMINS
# ============================================================
# Chemin vers le dataset New Plant Diseases Dataset
# Structure attendue :
#   DATASET_PATH/
#   ├── train/          (80% - ~70,200 images)
#   │   ├── Apple___Apple_scab/
#   │   ├── Apple___Black_rot/
#   │   └── ... (38 classes)
#   └── valid/          (20% - ~17,560 images)
#       ├── Apple___Apple_scab/
#       └── ... (38 classes)
DATASET_PATH = "data/New Plant Diseases Dataset(Augmented)/New Plant Diseases Dataset(Augmented)"

# Chemins de sortie
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
CHECKPOINT_DIR = os.path.join(BASE_DIR, "checkpoints")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Creer les repertoires s'ils n'existent pas
for d in [CHECKPOINT_DIR, RESULTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# ============================================================
# PARAMETRES D'ENTRAINEMENT
# ============================================================
IMG_SIZE = 224              # Taille des images (224x224 pour les modeles pre-entraines)
BATCH_SIZE = 32             # Taille du batch
EPOCHS = 50                 # Nombre maximum d'epochs
LEARNING_RATE = 0.0001      # Taux d'apprentissage initial
NUM_CLASSES = 38            # Nombre de classes (maladies + sain)

# ============================================================
# PARAMETRES DE PRETRAITEMENT
# ============================================================
# Normalisation ImageNet (pour le transfer learning)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# Data Augmentation
ROTATION_RANGE = 20         # Degres de rotation
ZOOM_RANGE = 0.2            # Facteur de zoom
HORIZONTAL_FLIP = True      # Flip horizontal
VERTICAL_FLIP = False       # Flip vertical (souvent desactive pour les feuilles)
BRIGHTNESS_RANGE = [0.8, 1.2]  # Plage de luminosite
SHEAR_RANGE = 0.1           # Cisaillement

# ============================================================
# ARCHITECTURES CNN A EVALUER
# ============================================================
# Liste des modeles a comparer (conformement a l'etat de l'art)
MODELS_TO_TRAIN = [
    "resnet50",      # ~25M params - Connexions residuelles
    "densenet121",   # ~8M params - Connexions denses
    "efficientnetb0",# ~5.3M params - Meilleur rapport precision/efficacite
    "mobilenetv2",   # ~3.5M params - Architecture legere (mobile)
]

# Modele par defaut si un seul est entraine
DEFAULT_MODEL = "efficientnetb0"

# ============================================================
# PARAMETRES DE CALLBACKS
# ============================================================
EARLY_STOPPING_PATIENCE = 10    # Epochs sans amelioration avant arret
REDUCE_LR_PATIENCE = 5          # Epochs sans amelioration avant reduction LR
REDUCE_LR_FACTOR = 0.5          # Facteur de reduction du LR
MIN_LR = 1e-7                   # LR minimum

# ============================================================
# PARAMETRES DE VALIDATION
# ============================================================
TEST_SPLIT = 0.2                # Fraction pour le test (si pas de dossier valid)
RANDOM_SEED = 42                # Graine pour la reproductibilite

# ============================================================
# PARAMETRES DE GRAD-CAM
# ============================================================
GRADCAM_LAYER = "conv5_block32_concat"  # Couche pour Grad-CAM (DenseNet)
NUM_GRADCAM_SAMPLES = 5         # Nombre d'echantillons pour Grad-CAM
