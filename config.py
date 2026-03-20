"""
Central configuration for the WikiArt artist classification project.

All paths, hyperparameters, and constants are defined here so that
every other module reads from a single source of truth.
"""

import os

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(PROJECT_ROOT, "wikiart")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "outputs")
MODEL_DIR = os.path.join(OUTPUT_DIR, "models")
PLOT_DIR = os.path.join(OUTPUT_DIR, "plots")
LOG_DIR = os.path.join(OUTPUT_DIR, "logs")

# Create output directories if they don't exist
for d in [OUTPUT_DIR, MODEL_DIR, PLOT_DIR, LOG_DIR]:
    os.makedirs(d, exist_ok=True)

# ──────────────────────────────────────────────
# Reproducibility
# ──────────────────────────────────────────────
SEED = 42

# ──────────────────────────────────────────────
# Data
# ──────────────────────────────────────────────
# Target image size after resizing (height, width)
IMG_SIZE = (224, 224)

# Fraction of data used for validation and test
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15  # remaining 70 % is training

# ──────────────────────────────────────────────
# Training
# ──────────────────────────────────────────────
BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-4

# Early-stopping patience (number of epochs without improvement)
EARLY_STOP_PATIENCE = 5

# Learning-rate reduction patience
LR_REDUCE_PATIENCE = 3
LR_REDUCE_FACTOR = 0.5

# ──────────────────────────────────────────────
# Class names (derived automatically from folder names)
# ──────────────────────────────────────────────
CLASS_NAMES = sorted(
    [d for d in os.listdir(DATA_DIR)
     if os.path.isdir(os.path.join(DATA_DIR, d))]
)
NUM_CLASSES = len(CLASS_NAMES)
