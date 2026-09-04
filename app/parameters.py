import os
from pathlib import Path
from dataclasses import dataclass

# --- Paths ---

BASE_DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
PROCESSED_DIR = BASE_DATA_DIR / "processed"
COORDS_PATH = PROCESSED_DIR / "clustered_data.npz"
METADATA_PATH = PROCESSED_DIR / "metadata.json"
MASTER_EMBEDDINGS_PATH = PROCESSED_DIR / "master_embeddings.npy"

# --- Emotional Embeddings ---

ATTENUATION_FACTOR = 0.1
LINE_EMOTION_THRESHOLD = 0.25

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
    ],
}


@dataclass
class Weights:
    emotional_weight: float
    line_weight: float
    context_line_amount: int
    neighborhood_amount: int
    max_results: int
