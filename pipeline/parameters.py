import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
BASE_DATA_DIR = Path(os.getenv("DATA_DIR", "data"))

RAW_DATA_DIR = BASE_DATA_DIR / "raw"
SEMI_PROCESSED_DIR = BASE_DATA_DIR / "semi-processed"
PROCESSED_DIR = BASE_DATA_DIR / "processed"
CONFIG_PATH = BASE_DATA_DIR / "config"
MODELS_DIR = BASE_DATA_DIR / "models"

# --- File Paths ---
RAW_POEMS_PATH = RAW_DATA_DIR / "poems.json"
MEMMAP_PATH = SEMI_PROCESSED_DIR / "raw_embeddings.dat"
METADATA_PATH = PROCESSED_DIR / "metadata.json"
CLUSTERED_DATA_PATH = PROCESSED_DIR / "clustered_data.npz"
MASTER_EMBEDDINGS_PATH = PROCESSED_DIR / "master_embeddings.npy"
EMOTIONAL_ANCHORS_PATH = CONFIG_PATH / "emotional-anchor.json"

# --- Model Artifact Paths ---
PCA_MODEL_PATH = MODELS_DIR / "pca_model.joblib"
UMAP_MODEL_PATH = MODELS_DIR / "umap_model.joblib"
HDBSCAN_MODEL_PATH = MODELS_DIR / "hdbscan_model.joblib"

# --- Build Database Parameters ---
LINE_LENGTH_THRESHOLD = 15
MAX_LINE_LENGTH = 250
WRITE_BATCH_SIZE = 2000
WORD_EMOTION_THRESHOLD = 0.51
LINE_EMOTION_THRESHOLD = 0.25
ATTENUATION_FACTOR = 0.2
EMBEDDER_MODEL = "all-MiniLM-L6-v2"
SPACY_MODEL = "en_core_web_sm"
LRU_CACHE_SIZE = 10000

# --- Generate Dataset Parameters ---
READ_BATCH_SIZE = 50000
SKIP = 0
EMBEDDING_DIM = 384
BATCH_FLUSH_SIZE = 10000  # Stream batch size for Neo4j extraction
TRAINING_SAMPLE_SIZE = 300000  # Cap on random sample for model training


class PCA_parameters:
    n_components = 50
    batch_size = 4096


class UMAP_parameters:
    n_components = 5
    n_neighbors = 10
    min_dist = 0.0
    metric = "cosine"
    init = "spectral"
    verbose = False
    low_memory = True
    n_jobs = -1


class HDBSCAN_parameters:
    min_cluster_size = 20
    min_samples = 5
    metric = "euclidean"
    n_jobs = -1
    prediction_data = True
