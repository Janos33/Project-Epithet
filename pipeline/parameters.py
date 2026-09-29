import json
import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

# --- Paths ---
BASE_DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
CURRENT_PROFILE = os.environ.get("PROFILE")

# --- Validation Check ---
if not CURRENT_PROFILE:
    raise RuntimeError(
        "ERROR: 'PROFILE' environment variable is not set.\n"
        "Please specify a profile when running Docker, e.g.:\n"
        "docker compose run --rm -e PROFILE=classic build-database"
    )

RAW_DATA_DIR = BASE_DATA_DIR / "raw"
TEMPORARY_DIR = BASE_DATA_DIR / "temporary"
PROFILE_DIR = BASE_DATA_DIR / "profiles" / CURRENT_PROFILE

CONFIG_DIR = PROFILE_DIR / "config"
PROCESSED_DIR = PROFILE_DIR / "processed"

TEMPORARY_DIR.mkdir(parents=True, exist_ok=True)
PROCESSED_DIR.mkdir(parents=True, exist_ok=True)

# --- File Paths ---
PROFILE_PATH = CONFIG_DIR / "profile.json"
RAW_POEMS_PATH = RAW_DATA_DIR / "poems.json"
MEMMAP_PATH = TEMPORARY_DIR / "raw_embeddings.dat"
METADATA_PATH = PROCESSED_DIR / "metadata.jsonl"
CLUSTERED_DATA_PATH = PROCESSED_DIR / "clustered_data.npz"
MASTER_EMBEDDINGS_PATH = PROCESSED_DIR / "master_embeddings.npy"
EMOTIONAL_ANCHORS_PATH = CONFIG_DIR / "emotional-anchor.json"

# --- Model Artifact Paths ---
PCA_MODEL_PATH = TEMPORARY_DIR / "pca_model.joblib"
UMAP_MODEL_PATH = TEMPORARY_DIR / "umap_model.joblib"
HDBSCAN_MODEL_PATH = TEMPORARY_DIR / "hdbscan_model.joblib"


# --- Dataset Profile (data/config/profile.json) ---
# Non-performance settings live in profile.json:
#   "shared"        - read by both the pipeline and the app; must match the built dataset
#   "pipeline_only" - only used here, when building the dataset
# Performance settings (batch sizes, cache sizes, job counts) stay in this file.
_SUPPORTED_SCHEMA_VERSION = 1


def _load_profile(path: Path) -> dict:
    try:
        with open(path, "r", encoding="utf-8") as f:
            profile = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset profile not found: {path}") from None

    version = profile.get("schema_version")
    if version != _SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            f"{path} has schema_version {version!r}, expected {_SUPPORTED_SCHEMA_VERSION}"
        )
    return profile


_profile = _load_profile(PROFILE_PATH)
_shared = _profile["shared"]
_pipeline = _profile["pipeline_only"]

# --- Shared with the app (profile.json -> "shared") ---
EMBEDDER_MODEL = _shared["embedding"]["model"]
EMBEDDING_DIM = _shared["embedding"]["dim"]

LINE_LENGTH_THRESHOLD = _shared["chunking"]["line_length_threshold"]
MAX_LINE_LENGTH = _shared["chunking"]["max_line_length"]

LINE_EMOTION_ATTENUATION_THRESHOLD = _shared["line_scoring"][
    "emotion_attenuation_threshold"
]
LINE_ATTENUATION_FACTOR = _shared["line_scoring"]["attenuation_factor"]
LINE_EXCLUDE_ATTENUATION_FACTOR = _shared["line_scoring"]["exclude_attenuation_factor"]

# --- Pipeline only (profile.json -> "pipeline_only") ---
SPACY_MODEL = _pipeline["spacy_model"]

POEMS_TO_PROCESS = _pipeline["poems_to_process"]  # 0 processes all poems

LINE_EMOTION_THRESHOLD = _pipeline["line_filter"]["emotion_threshold"]

WORD_EMOTION_THRESHOLD = _pipeline["word_scoring"]["emotion_threshold"]
WORD_EMOTION_ATTENUATION_THRESHOLD = _pipeline["word_scoring"][
    "emotion_attenuation_threshold"
]
WORD_ATTENUATION_FACTOR = _pipeline["word_scoring"]["attenuation_factor"]
WORD_EXCLUDE_ATTENUATION_FACTOR = _pipeline["word_scoring"][
    "exclude_attenuation_factor"
]

OVERALL_SCORE_THRESHOLD = _pipeline["overall_score_threshold"]
WORD_DELETE_THRESHOLD = _pipeline["word_delete_threshold"]

COLOR_AMOUNT = _pipeline["word_color_scoring"]["color_amount"]
COLOR_ATTENUATION_FACTOR = _pipeline["word_color_scoring"]["color_attenuation_factor"]

_clustering = _pipeline["clustering"]

# --- Performance settings (stay in code) ---
WRITE_BATCH_SIZE = 2000
LRU_CACHE_SIZE = 10000
BATCH_FLUSH_SIZE = 10000  # Stream batch size for Neo4j extraction
TRAINING_SAMPLE_SIZE = 300000  # Cap on random sample for model training


class PCA_parameters:
    n_components = _clustering["pca"]["n_components"]
    batch_size = 4096  # performance


class UMAP_parameters:
    n_components = _clustering["umap"]["n_components"]
    n_neighbors = _clustering["umap"]["n_neighbors"]
    min_dist = _clustering["umap"]["min_dist"]
    metric = _clustering["umap"]["metric"]
    init = _clustering["umap"]["init"]
    # performance
    verbose = False
    low_memory = True
    n_jobs = -1


class HDBSCAN_parameters:
    min_cluster_size = _clustering["hdbscan"]["min_cluster_size"]
    min_samples = _clustering["hdbscan"]["min_samples"]
    metric = _clustering["hdbscan"]["metric"]
    # performance
    n_jobs = -1
    # required by hdbscan.approximate_predict in generate_dataset.py
    prediction_data = True
