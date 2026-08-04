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
EMOTION_THRESHOLD = 0.5

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
        "sunshine",
        "cupcake",
    ],
    "Serenity": [
        "stillness",
        "silence",
        "calm",
        "infinity",
        "cosmos",
        "eternal",
        "wonder",
        "breeze",
        "timeless",
        "awe",
    ],
    "Passion": [
        "passion",
        "desire",
        "burning",
        "obsession",
        "tender",
        "heartbeat",
        "yearning",
    ],
    "Melancholy": [
        "grief",
        "loneliness",
        "mourning",
        "regret",
        "guilt",
        "shame",
        "sorrow",
        "yesterday",
        "absence",
        "memory",
        "bittersweet",
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
        "ruin",
        "unfair",
        "unjust",
    ],
    "Delirium": [
        "madness",
        "insanity",
        "frenzy",
        "chaos",
        "eerie",
        "haunting",
        "shadow",
        "disgust",
        "abyss",
        "mystery",
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
        "wither",
        "earth",
        "physical",
    ],
}

COLOR_MAP = {
    "Neutral": "#A0A0A0",
    "Radiance": "#F4D35E",
    "Serenity": "#3B7A57",
    "Passion": "#9B1D20",
    "Melancholy": "#3D5A80",
    "Torment": "#D95D39",
    "Delirium": "#5E3A6D",
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