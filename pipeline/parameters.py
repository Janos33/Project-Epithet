# pipeline/params.py (Committed to Git)
import os
from pathlib import Path

# --- Paths ---
# Reads DATA_DIR from .env if present, otherwise defaults to "data"
BASE_DATA_DIR = Path(os.getenv("DATA_DIR", "data"))

RAW_DATA_DIR = BASE_DATA_DIR / "raw"
SEMI_PROCESSED_DIR = BASE_DATA_DIR / "semi-processed"
PROCESSED_DIR = BASE_DATA_DIR / "processed"

RAW_POEMS_PATH = RAW_DATA_DIR / "poems.json"
EXTRACTED_COORDS_PATH = SEMI_PROCESSED_DIR / "coords.npz"
METADATA_PATH = PROCESSED_DIR / "metadata.json"
CLUSTERED_DATA_PATH = PROCESSED_DIR / "clustered_data.npz"
MEMMAP_PATH = SEMI_PROCESSED_DIR / "raw_embeddings.dat"

# --- Database Parameters ---
LINE_LENGTH_THRESHOLD = 15
MAX_LINE_LENGTH = 50

# Can read from .env with a fallback default
WRITE_BATCH_SIZE = 2000
WORD_EMOTION_THRESHOLD = 0.51
LINE_EMOTION_THRESHOLD = 0.25
SCORE_SIMILARITY_THRESHOLD = 0.1

ATTENUATION_FACTOR = 0.1

EMBEDDER_MODEL = "all-MiniLM-L6-v2"
SPACY_MODEL = "en_core_web_sm"
LRU_CACHE_SIZE = 10000

# --- Emotional Anchors & Colors ---
EMOTIONAL_ANCHORS = {
    "Radiance": [
        "glow",
        "radiance",
        "joy",
        "laughter",
        "hope",
        "dawn",
        "promise",
        "blossom",
        "sunshine"
    ],
    "Serenity": [
        "stillness",
        "silence",
        "calm",
        "infinity",
        "eternal",
        "wonder",
        "breeze",
        "timeless",
        "awe"
    ],
    "Passion": [
        "passion",
        "desire",
        "burning",
        "obsession",
        "tender",
        "heartbeat",
        "yearning"
    ],
    "Melancholy": [
        "grief",
        "loneliness",
        "mourning",
        "regret",
        "guilt",
        "shame",
        "sorrow",
        "absence",
        "memory",
        "bittersweet"
    ],
    "Torment": [
        "dread",
        "fear",
        "rage",
        "storm",
        "shatter",
        "pain",
        "broken",
        "desolation",
        "ruin"
    ],
    "Delirium": [
        "madness",
        "insanity",
        "frenzy",
        "chaos",
        "eerie",
        "haunting",
        "shadow",
        "abyss",
        "mystery"
    ],
    "Transience": [
        "time",
        "fading",
        "dust",
        "fleeting",
        "autumn",
        "mortality",
        "ephemeral",
        "passing",
        "vanishing",
        "wither"
    ],
}

COLOR_MAP = {
    "Neutral": "#A0A0A0",
    "Radiance": "#F4D35E",
    "Serenity": "#389261",
    "Passion": "#641012",
    "Melancholy": "#7BA1D1",
    "Torment": "#D95D39",
    "Delirium": "#462E86",
    "Transience": "#8B4513",
}

# --- Dataset Generation Parameters ---
READ_BATCH_SIZE = 50000
SKIP = 0
CURRENT_IDX = 0
EMBEDDING_DIM = 384


class PCA_parameters:
    n_components = 50
    random_state = 42


class UMAP_parameters:
    n_components = 10
    n_neighbors = 15
    min_dist = 0.1
    metric = "cosine"
    init = "random"
    verbose = True


class HDBSCAN_parameters:
    min_cluster_size = 15
    min_samples = 5
    metric = "euclidean"