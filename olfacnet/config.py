"""
Centralized configuration and hyperparameters for OlfacNet v2.
"""

import os
import torch

# ─── Paths ───────────────────────────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
RAW_DIR = os.path.join(DATA_DIR, "raw")
PROCESSED_DIR = os.path.join(DATA_DIR, "processed")
GRAPHS_DIR = os.path.join(DATA_DIR, "graphs")
SPECTRA_DIR = os.path.join(PROCESSED_DIR, "spectra")
EXPERIMENTS_DIR = os.path.join(BASE_DIR, "experiments")
CHECKPOINTS_DIR = os.path.join(EXPERIMENTS_DIR, "checkpoints")
RESULTS_DIR = os.path.join(EXPERIMENTS_DIR, "results")

# ─── Device ──────────────────────────────────────────────────────────────────
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── Data ────────────────────────────────────────────────────────────────────
NUM_CLASSES = 105            # Number of odor categories
MAX_ATOMS = 50               # Max atoms per molecule graph
ATOM_FEATURE_DIM = 1         # Node feature dimension in pre-computed graphs
MORGAN_FP_BITS = 2048        # Morgan fingerprint length
MORGAN_RADIUS = 2            # Morgan fingerprint radius
TRAIN_SPLIT = 0.8            # Train/val split ratio
VAL_SPLIT = 0.1              # Validation split ratio
TEST_SPLIT = 0.1             # Test split ratio
RANDOM_SEED = 42

# ─── Model Architecture ─────────────────────────────────────────────────────
GNN_HIDDEN_DIM = 128         # Hidden dim for GCN layers
GNN_OUTPUT_DIM = 128         # Output dim of GNN after pooling
GNN_NUM_LAYERS = 3           # Number of GCN layers
GNN_DROPOUT = 0.3            # Dropout in GNN

FP_HIDDEN_DIM = 512          # Hidden dim for fingerprint encoder
FP_OUTPUT_DIM = 128          # Output dim of fingerprint branch

ATTENTION_DIM = 256          # Attention fusion dimension
ATTENTION_HEADS = 4          # Number of attention heads

CLASSIFIER_HIDDEN = 256      # Classifier hidden layer
CLASSIFIER_DROPOUT = 0.4     # Classifier dropout

# ─── Training ────────────────────────────────────────────────────────────────
BATCH_SIZE = 64
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS = 30
PATIENCE = 7                 # Early stopping patience
LR_SCHEDULER_FACTOR = 0.5    # ReduceLROnPlateau factor
LR_SCHEDULER_PATIENCE = 3    # ReduceLROnPlateau patience
GRADIENT_CLIP_NORM = 1.0     # Max gradient norm

# Focal Loss
FOCAL_ALPHA = 0.25
FOCAL_GAMMA = 2.0

# ─── Inference ───────────────────────────────────────────────────────────────
TOP_K_PREDICTIONS = 10       # Number of top predictions to return
CONFIDENCE_THRESHOLD = 0.3   # Min confidence to report a prediction
