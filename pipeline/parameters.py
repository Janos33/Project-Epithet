import os
from pathlib import Path

# --- Paths ---
BASE_DATA_DIR = Path(os.getenv("DATA_DIR", "data"))

RAW_DATA_DIR = BASE_DATA_DIR / "raw"
SEMI_PROCESSED_DIR = BASE_DATA_DIR / "semi-processed"
PROCESSED_DIR = BASE_DATA_DIR / "processed"
CONFIG_PATH = BASE_DATA_DIR / "config"

RAW_POEMS_PATH = RAW_DATA_DIR / "poems.json"
EXTRACTED_COORDS_PATH = SEMI_PROCESSED_DIR / "coords.npz"
METADATA_PATH = PROCESSED_DIR / "metadata.json"
CLUSTERED_DATA_PATH = PROCESSED_DIR / "clustered_data.npz"
MEMMAP_PATH = SEMI_PROCESSED_DIR / "raw_embeddings.dat"
MASTER_EMBEDDINGS_PATH = PROCESSED_DIR / "master_embeddings.npy"
EMOTIONAL_ANCHORS_PATH = CONFIG_PATH / "emotional-anchor.json"

# --- Build Database Parameters ---
LINE_LENGTH_THRESHOLD = 15
MAX_LINE_LENGTH = 250

WRITE_BATCH_SIZE = 2000
WORD_EMOTION_THRESHOLD = 0.35

LINE_EMOTION_THRESHOLD = 0.35
ATTENUATION_FACTOR = 0.2

EMBEDDER_MODEL = "all-MiniLM-L6-v2"
SPACY_MODEL = "en_core_web_sm"
LRU_CACHE_SIZE = 10000

# --- Generate Dataset Parameters ---
READ_BATCH_SIZE = 50000
SKIP = 0
EMBEDDING_DIM = 384


class PCA_parameters:
    n_components = 50
    random_state = 42


class UMAP_parameters:
    n_components = 5
    n_neighbors = 10
    min_dist = 0.0
    metric = "cosine"
    init = "spectral"
    verbose = True


class HDBSCAN_parameters:
    min_cluster_size = 20
    min_samples = 5
    metric = "euclidean"
