import json
import os
from pathlib import Path
from dataclasses import dataclass

# --- Paths ---
# The app can serve several datasets ("profiles") at once, each its own
# folder: data/profiles/<name>/{config,processed}. Which ones are actually
# loaded is decided in main.py, not here -- this module only knows how to
# list what's on disk and how to load one profile's config.
BASE_DATA_DIR = Path(os.getenv("DATA_DIR", "data"))
PROFILES_DIR = BASE_DATA_DIR / "profiles"

_SUPPORTED_SCHEMA_VERSION = 1


def list_profiles() -> list[str]:
    """Names of profile folders under data/profiles/ that have a config/profile.json."""
    if not PROFILES_DIR.is_dir():
        return []
    return sorted(
        p.name
        for p in PROFILES_DIR.iterdir()
        if (p / "config" / "profile.json").is_file()
    )


@dataclass(frozen=True)
class ProfileConfig:
    name: str
    display_name: str
    description: str

    emotional_anchors_path: Path
    master_embeddings_path: Path
    metadata_path: Path
    clustered_data_path: Path

    # --- Parameters (profile.json -> "shared") ---
    # embedder_model / embedding_dim   - Model the dataset was built with, and its vector size.
    # line_emotion_attenuation_threshold - Line emotion scores below this are weakened.
    # line_attenuation_factor          - Multiplier applied to those weak scores.
    # line_exclude_attenuation_factor  - How strongly "exclude" phrases push away from an emotion anchor.
    embedder_model: str
    embedding_dim: int
    line_emotion_attenuation_threshold: float
    line_attenuation_factor: float
    line_exclude_attenuation_factor: float


def load_profile(name: str) -> ProfileConfig:
    """
    Load one profile's config from data/profiles/<name>/config/profile.json.

    Only the "shared" section is used here -- "pipeline_only" values are for
    the pipeline and are ignored by the app. Raises FileNotFoundError,
    ValueError, or KeyError on a missing, mismatched, or malformed profile,
    so a broken profile fails loudly at load time instead of serving quietly
    wrong results later.
    """
    profile_dir = PROFILES_DIR / name
    config_path = profile_dir / "config"
    processed_path = profile_dir / "processed"
    profile_path = config_path / "profile.json"

    try:
        with open(profile_path, "r", encoding="utf-8") as f:
            data = json.load(f)
    except FileNotFoundError:
        raise FileNotFoundError(f"Dataset profile not found: {profile_path}") from None

    version = data.get("schema_version")
    if version != _SUPPORTED_SCHEMA_VERSION:
        raise ValueError(
            f"{profile_path} has schema_version {version!r}, "
            f"expected {_SUPPORTED_SCHEMA_VERSION}"
        )

    shared = data["shared"]
    line_scoring = shared["line_scoring"]

    return ProfileConfig(
        name=name,
        display_name=data.get("name", name),
        description=data.get("description", ""),
        emotional_anchors_path=config_path / "emotional-anchor.json",
        master_embeddings_path=processed_path / "master_embeddings.npy",
        metadata_path=processed_path / "metadata.json",
        clustered_data_path=processed_path / "clustered_data.npz",
        embedder_model=shared["embedding"]["model"],
        embedding_dim=shared["embedding"]["dim"],
        line_emotion_attenuation_threshold=line_scoring[
            "emotion_attenuation_threshold"
        ],
        line_attenuation_factor=line_scoring["attenuation_factor"],
        line_exclude_attenuation_factor=line_scoring["exclude_attenuation_factor"],
    )


# --- Dataclass for forwarding weights ---


@dataclass
class Weights:
    emotional_weight: float
    line_weight: float
    context_line_amount: int
    neighborhood_amount: int
    max_results: int
