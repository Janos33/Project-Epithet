import os
from pathlib import Path
from dataclasses import dataclass

# Note: ensure that the data here is consistent with the state that pipeline/parameters.py was in when the pipeline was last run.

# --- Paths ---

BASE_DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
PROCESSED_DIR = BASE_DATA_DIR / "processed"
COORDS_PATH = PROCESSED_DIR / "clustered_data.npz"
METADATA_PATH = PROCESSED_DIR / "metadata.json"
CONFIG_PATH = BASE_DATA_DIR / "config"
MASTER_EMBEDDINGS_PATH = PROCESSED_DIR / "master_embeddings.npy"
EMOTIONAL_ANCHORS_PATH = CONFIG_PATH / "emotional-anchor.json"

# --- Parameters ---
# LINE_EMOTION_THRESHOLD - Minimum average emotion score for a line to be considered emotionally significant.
# ATTENUATION_FACTOR - Weakens the influence of emotions that are below the LINE_EMOTION_THRESHOLD so words have stronger identities.

LINE_EMOTION_THRESHOLD = 0.1
ATTENUATION_FACTOR = 0.2
EMBEDDING_DIM = 384

# --- Dataclass for forwarding weights ---


@dataclass
class Weights:
    emotional_weight: float
    line_weight: float
    context_line_amount: int
    neighborhood_amount: int
    max_results: int
