import json
import os
from pathlib import Path
from dataclasses import dataclass

# --- Paths ---
BASE_DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
PROCESSED_DIR = BASE_DATA_DIR / "processed"
CONFIG_PATH = BASE_DATA_DIR / "config"

PROFILE_PATH = CONFIG_PATH / "profile.json"
CLUSTERED_DATA_PATH = PROCESSED_DIR / "clustered_data.npz"
METADATA_PATH = PROCESSED_DIR / "metadata.json"
MASTER_EMBEDDINGS_PATH = PROCESSED_DIR / "master_embeddings.npy"
EMOTIONAL_ANCHORS_PATH = CONFIG_PATH / "emotional-anchor.json"

# --- Dataset Profile (data/config/profile.json) ---
# The profile must match the version of the code it was build for.
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
            f"File '{path}' uses schema version {version!r}, "
            f"but this code supports schema version {_SUPPORTED_SCHEMA_VERSION!r}."
        )
    return profile


_shared = _load_profile(PROFILE_PATH)["shared"]

# --- Parameters (profile.json -> "shared") ---
# EMBEDDER_MODEL / EMBEDDING_DIM - Embedding model the dataset was built with, and its vector size.
# LINE_EMOTION_ATTENUATION_THRESHOLD - Line emotion scores below this are weakened.
# LINE_ATTENUATION_FACTOR - Multiplier applied to those weak scores.
# LINE_EXCLUDE_ATTENUATION_FACTOR - How strongly "exclude" phrases push away from an emotion anchor.
EMBEDDER_MODEL = _shared["embedding"]["model"]
EMBEDDING_DIM = _shared["embedding"]["dim"]

LINE_EMOTION_ATTENUATION_THRESHOLD = _shared["line_scoring"][
    "emotion_attenuation_threshold"
]
LINE_ATTENUATION_FACTOR = _shared["line_scoring"]["attenuation_factor"]
LINE_EXCLUDE_ATTENUATION_FACTOR = _shared["line_scoring"]["exclude_attenuation_factor"]

# --- Dataclass for forwarding weights ---


@dataclass
class Weights:
    emotional_weight: float
    line_weight: float
    context_line_amount: int
    neighborhood_amount: int
    max_results: int
