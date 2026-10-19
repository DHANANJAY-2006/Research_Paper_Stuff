"""
config.py — Configuration variables and paths for the Adversarial IDS project.
"""

import os

# ==========================================
# PATHS
# ==========================================
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_RAW_DIR = os.path.join(PROJECT_ROOT, "data", "raw")
DATA_PROCESSED_DIR = os.path.join(PROJECT_ROOT, "data", "processed")
MODELS_DIR = os.path.join(PROJECT_ROOT, "results", "models")
FIGURES_DIR = os.path.join(PROJECT_ROOT, "results", "figures")

# Ensure directories exist
os.makedirs(DATA_RAW_DIR, exist_ok=True)
os.makedirs(DATA_PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# ==========================================
# DATASET SETTINGS (CIC-IoT2023)
# ==========================================
# Benign class label in CIC-IoT2023
BENIGN_LABEL = "BenignTraffic"
TARGET_COLUMN = "label"

# We will sample the dataset to avoid memory crashes on local machines.
MAX_ROWS = 500000 
TEST_SIZE = 0.2
RANDOM_SEED = 42

# Features that CAN be manipulated by an attacker (Problem-Space Constraints)
MUTATABLE_FEATURES = [
    "Duration",
    "Tot size of bckwd packet",
    "Tot size of fwd packet",
    "Fwd Packet Length Max",
    "Bckwd Packet Length Max",
    "Flow IAT Mean",
    "Bckwd IAT Mean"
]

# Features that CANNOT be changed without breaking the TCP/IP protocol
IMMUTABLE_FEATURES = [
    "Protocol",
    "Port",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count"
]
