# Detection des Maladies des Plantes par CNN

Projet de Fin d'Etudes - Vision par ordinateur et Deep Learning

## Description

Ce projet implemente un systeme de detection automatique des maladies des plantes a partir d'images de feuilles, base sur des reseaux de neurones convolutifs (CNN) avec transfer learning. Quatre architectures modernes sont evaluees : ResNet-50, DenseNet-121, EfficientNet-B0 et MobileNetV2.

## Structure du Projet

```
plant_disease_project/
|
|-- config.py                  # Configuration centrale (parametres modifiables)
|-- requirements.txt           # Dependances Python
|-- README.md                  # Ce fichier
|
|-- 1_preprocessing.py         # Phase 1 : Pretraitement des donnees
|-- 2_cnn_models.py            # Phase 2 : Developpement des architectures CNN
|-- 3_training.py              # Phase 3 : Entrainement des modeles
|-- 4_validation.py            # Phase 4 : Validation et evaluation
|-- 5_analysis.py              # Phase 5 : Analyse avancee (Grad-CAM, etc.)
|
|-- utils/
|   |-- __init__.py
|   |-- data_utils.py          # Utilitaires de chargement des donnees
|   |-- model_utils.py         # Utilitaires de construction des modeles
|   |-- viz_utils.py           # Utilitaires de visualisation
|
|-- data/                      # Dataset (a telecharger)
|-- checkpoints/               # Modeles sauvegardes
|-- models/                    # Architectures sauvegardees
|-- results/                   # Resultats et visualisations
```

## Installation

### 1. Cloner et configurer l'environnement

```bash
# Creer un environnement virtuel
python -m venv venv

# Activer l'environnement
source venv/bin/activate        # Linux/Mac
venv\Scripts\activate           # Windows

# Installer les dependances
pip install -r requirements.txt
```

### 2. Telecharger le Dataset

```bash
# Via Kaggle API ( necessite un compte Kaggle )
kaggle datasets download -d vipoooool/new-plant-diseases-dataset
unzip new-plant-diseases-dataset.zip -d data/
```

Ou telechargez manuellement depuis :
https://www.kaggle.com/datasets/vipoooool/new-plant-diseases-dataset

### 3. Verifier la structure

```
data/New Plant Diseases Dataset(Augmented)/
└── New Plant Diseases Dataset(Augmented)/
    ├── train/          # ~70,200 images (80%)
    │   ├── Apple___Apple_scab/
    │   ├── Apple___Black_rot/
    │   └── ... (38 classes)
    └── valid/          # ~17,560 images (20%)
        ├── Apple___Apple_scab/
        └── ... (38 classes)
```

## Utilisation

### Execution sequentielle ( recommandee )

```bash
# Phase 1 : Pretraitement
python 1_preprocessing.py

# Phase 2 : Construction des modeles
python 2_cnn_models.py

# Phase 3 : Entrainement (modele par defaut : EfficientNet-B0)
python 3_training.py

# Phase 4 : Validation
python 4_validation.py

# Phase 5 : Analyse avancee (Grad-CAM, etc.)
python 5_analysis.py
```

### Exemples d'utilisation avancee

```bash
# Entrainer un modele specifique
python 3_training.py --model resnet50

# Entrainer avec des hyperparametres personnalises
python 3_training.py --model densenet121 --epochs 30 --lr 0.0005 --batch-size 16

# Activer le fine-tuning (defrost couches de base)
python 3_training.py --model efficientnetb0 --fine-tune

# Entrainer tous les modeles
python 3_training.py --all

# Evaluer un modele specifique
python 4_validation.py --model mobilenetv2

# Evaluer tous les modeles
python 4_validation.py --all

# Generer uniquement les visualisations Grad-CAM
python 5_analysis.py --model resnet50 --gradcam

# Comparer tous les modeles
python 5_analysis.py --compare-all
```

## Phases du Projet

### Phase 1 : Pretraitement
- Chargement des images depuis le dataset
- Data augmentation (rotation, zoom, flip, luminosite)
- Normalisation des pixels (1/255)
- Creation des generateurs train/validation
- Statistiques du dataset

### Phase 2 : Developpement CNN
- 4 architectures avec transfer learning depuis ImageNet :
  - **ResNet-50** (~25M params) : Connexions residuelles
  - **DenseNet-121** (~8M params) : Connexions denses
  - **EfficientNet-B0** (~5.3M params) : Meilleur rapport precision/efficacite
  - **MobileNetV2** (~3.5M params) : Architecture legere pour mobile

### Phase 3 : Entrainement
- Transfer learning (couches de base gelees)
- Callbacks : Early stopping, Reduce LR, Model checkpoint, CSV logger, TensorBoard
- Mixed precision (acceleration GPU)
- Option de fine-tuning

### Phase 4 : Validation
- Accuracy, Precision, Recall, F1-score
- Top-3 Accuracy
- Matrice de confusion
- Rapport de classification par classe

### Phase 5 : Analyse
- **Grad-CAM** : Visualisation des zones d'interet du CNN
- Analyse des erreurs : Classes les plus confondues
- Comparaison multi-modeles
- Export des resultats pour le rapport

## Configuration

Les parametres principaux sont dans `config.py` :

| Parametre | Valeur par defaut | Description |
|-----------|------------------|-------------|
| IMG_SIZE | 224 | Taille des images |
| BATCH_SIZE | 32 | Taille des batches |
| EPOCHS | 50 | Nombre maximum d'epochs |
| LEARNING_RATE | 0.0001 | Taux d'apprentissage |
| NUM_CLASSES | 38 | Nombre de classes |

## Resultats Attendus

Fichiers generes dans `results/` :
- `{model}_training_curves.png` : Courbes d'entrainement
- `{model}_confusion_matrix.png` : Matrice de confusion
- `{model}_sample_predictions.png` : Exemples de predictions
- `{model}_class_performance.png` : Performances par classe
- `{model}_gradcam.png` : Visualisations Grad-CAM
- `{model}_error_analysis.png` : Analyse des erreurs
- `model_comparison.png` : Comparaison des architectures
- `final_results.json` : Resultats complets pour le rapport

## References de l'Etat de l'Art

| Auteur | Architecture | Precision |
|--------|-------------|-----------|
| Mohanty et al. (2016) | GoogLeNet + TL | 99.35% |
| Ferentinos (2018) | VGG + TL | 99.53% |
| Brahimi et al. (2018) | InceptionV3 + TL | 99.76% |
| Brahimi et al. (2018) | DenseNet-169 + TL | 99.72% |

## Auteur

Projet de Fin d'Etudes - Detection des maladies des plantes par vision par ordinateur
