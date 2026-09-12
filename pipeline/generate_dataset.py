import os
import gc
import json
import joblib
import numpy as np
import umap
import datetime

import hdbscan
from sklearn.decomposition import PCA
from neo4j import GraphDatabase, Driver
from dotenv import load_dotenv

load_dotenv()

# Note: ensure that the data here is consistent with the state that app/parameters.py will be once it is run.

from parameters import *


def get_neo4j_driver() -> Driver:
    return GraphDatabase.driver(
        os.getenv("NEO4J_URI"),
        auth=(os.getenv("NEO4J_USER"), os.getenv("NEO4J_PASSWORD")),
    )


def extract_data() -> int:
    """Extracts raw embeddings from Neo4j and saves them to disk."""
    print("Connecting to driver...")
    driver = get_neo4j_driver()

    with driver.session() as session:
        print("Counting total lines for memory allocation...")
        total_count = session.execute_read(
            lambda tx: tx.run("MATCH (n:Line) RETURN count(n) AS cnt").single()["cnt"]
        )

        projected_gb = (total_count * EMBEDDING_DIM * 4) / (1024**3)
        use_memmap = projected_gb >= 4.0

        if use_memmap:
            print(f"Dataset requires {projected_gb:.2f} GB. Using disk-backed memmap.")
            raw_embeddings = np.memmap(
                MEMMAP_PATH,
                dtype="float32",
                mode="w+",
                shape=(total_count, EMBEDDING_DIM),
            )
        else:
            print(f"Dataset requires {projected_gb:.2f} GB. Allocating in RAM.")
            raw_embeddings = np.empty((total_count, EMBEDDING_DIM), dtype="float32")

        current_idx = 0
        embedding_buffer = []
        metadata_buffer = []

        print(f"Streaming data from Neo4j to NDJSON ({METADATA_PATH})...")

        query = """
        MATCH (n:Line)
        RETURN n.text AS lineText, 
               n.embedding AS lineEmbedding,
               n.score AS score,
               [(n)-[r:HAS_KEYWORD]->(k:PoemKeyword) | {
                   word: k.text, 
                   color: r.color, 
                   score: r.score, 
                   inverseFrequency: k.inverseFrequency
               }] AS keywords
        """

        result = session.run(query, fetch_size=BATCH_FLUSH_SIZE)

        with open(METADATA_PATH, "wb") as f:
            for record in result:
                embedding_buffer.append(record["lineEmbedding"])

                lineData = {
                    "lineText": record["lineText"],
                    "keywords": record["keywords"],
                    "score": record["score"],
                }
                metadata_buffer.append(json.dumps(lineData, ensure_ascii=False))

                if len(embedding_buffer) >= BATCH_FLUSH_SIZE:
                    end_idx = current_idx + len(embedding_buffer)
                    raw_embeddings[current_idx:end_idx] = np.array(
                        embedding_buffer, dtype=np.float32
                    )

                    f.write(("\n".join(metadata_buffer) + "\n").encode("utf-8"))

                    current_idx = end_idx
                    embedding_buffer.clear()
                    metadata_buffer.clear()
                    print(
                        f"[{datetime.datetime.now()}] Extracted {current_idx} / {total_count} lines..."
                    )

            if embedding_buffer:
                end_idx = current_idx + len(embedding_buffer)
                raw_embeddings[current_idx:end_idx] = np.array(
                    embedding_buffer, dtype=np.float32
                )
                f.write(("\n".join(metadata_buffer) + "\n").encode("utf-8"))
                current_idx = end_idx
                embedding_buffer.clear()
                metadata_buffer.clear()

    driver.close()

    if use_memmap:
        raw_embeddings.flush()

    if not use_memmap:
        print(f"Saving binary embeddings array to {MASTER_EMBEDDINGS_PATH}...")
        raw_embeddings[:current_idx].tofile(MASTER_EMBEDDINGS_PATH)
    else:
        print(f"Renaming memmap file to {MASTER_EMBEDDINGS_PATH}...")
        if MEMMAP_PATH.exists():
            if MASTER_EMBEDDINGS_PATH.exists():
                MASTER_EMBEDDINGS_PATH.unlink()
            MEMMAP_PATH.rename(MASTER_EMBEDDINGS_PATH)

    print("Data extraction complete!")
    return current_idx


def train(total_idx: int):
    """Fits PCA, UMAP, and HDBSCAN on a sub-sample of data and saves the models."""
    print("\n--- Initiating Training Phase ---")

    print(f"Loading raw embeddings from {MASTER_EMBEDDINGS_PATH}...")
    embeddings = np.memmap(
        MASTER_EMBEDDINGS_PATH,
        dtype="float32",
        mode="r",
        shape=(total_idx, EMBEDDING_DIM),
    )

    sample_size = min(TRAINING_SAMPLE_SIZE, total_idx)
    sample_indices = np.random.choice(total_idx, sample_size, replace=False)

    # 1. Fit PCA
    print(f"Fitting PCA on {sample_size} samples...")
    pca = PCA(n_components=PCA_parameters.n_components)
    pca.fit(embeddings[sample_indices])
    joblib.dump(pca, PCA_MODEL_PATH)

    # Transform sample for UMAP
    pca_sample = pca.transform(embeddings[sample_indices])

    # 2. Fit UMAP
    print("Fitting UMAP on PCA output...")
    reducer = umap.UMAP(
        n_components=UMAP_parameters.n_components,
        n_neighbors=UMAP_parameters.n_neighbors,
        min_dist=UMAP_parameters.min_dist,
        metric=UMAP_parameters.metric,
        init=UMAP_parameters.init,
        verbose=UMAP_parameters.verbose,
        low_memory=UMAP_parameters.low_memory,
        n_jobs=UMAP_parameters.n_jobs,
    )
    reducer.fit(pca_sample)
    joblib.dump(reducer, UMAP_MODEL_PATH)

    # Transform sample for HDBSCAN
    umap_sample = reducer.transform(pca_sample)

    # 3. Fit HDBSCAN
    print("Fitting HDBSCAN clustering on UMAP output...")
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=HDBSCAN_parameters.min_cluster_size,
        min_samples=HDBSCAN_parameters.min_samples,
        metric=HDBSCAN_parameters.metric,
        core_dist_n_jobs=HDBSCAN_parameters.n_jobs,
        prediction_data=HDBSCAN_parameters.prediction_data,
    )
    clusterer.fit(umap_sample)
    joblib.dump(clusterer, HDBSCAN_MODEL_PATH)

    valid_clusters = len(set(clusterer.labels_)) - (1 if -1 in clusterer.labels_ else 0)
    print(
        f"Training complete. Found {valid_clusters} clusters in the sample. Models saved to {MODELS_DIR}."
    )

    del embeddings, pca_sample, umap_sample
    gc.collect()


def predict(total_idx: int):
    """Loads saved models and transforms/predicts clusters for the full dataset in batches."""
    print("\n--- Initiating Inference/Prediction Phase ---")

    print("Loading pre-trained models...")
    pca = joblib.load(PCA_MODEL_PATH)
    reducer = joblib.load(UMAP_MODEL_PATH)
    clusterer = joblib.load(HDBSCAN_MODEL_PATH)

    print(f"Streaming raw embeddings from {MASTER_EMBEDDINGS_PATH}...")
    embeddings = np.memmap(
        MASTER_EMBEDDINGS_PATH,
        dtype="float32",
        mode="r",
        shape=(total_idx, EMBEDDING_DIM),
    )

    # Pre-allocate output arrays
    umap_coords = np.empty((total_idx, UMAP_parameters.n_components), dtype="float32")
    cluster_labels = np.empty(total_idx, dtype="int32")

    batch_size = getattr(PCA_parameters, "batch_size", 10000)

    for i in range(0, total_idx, batch_size):
        end_idx = min(i + batch_size, total_idx)

        # Sequentially pass batch through the pipeline
        batch_pca = pca.transform(embeddings[i:end_idx])
        batch_umap = reducer.transform(batch_pca)
        batch_labels, _ = hdbscan.approximate_predict(clusterer, batch_umap)

        umap_coords[i:end_idx] = batch_umap
        cluster_labels[i:end_idx] = batch_labels

        print(f"[{datetime.datetime.now()}] Predicted {end_idx} / {total_idx} lines...")

    print(f"Saving clustered data to {CLUSTERED_DATA_PATH}...")
    np.savez_compressed(
        CLUSTERED_DATA_PATH, coords=umap_coords, cluster_labels=cluster_labels
    )

    del embeddings, umap_coords, cluster_labels
    gc.collect()
    print("Prediction phase complete!")


def main():
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    SEMI_PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # Phase 1: Extract
    if not MASTER_EMBEDDINGS_PATH.is_file() or not METADATA_PATH.is_file():
        total_idx = extract_data()
    else:
        print("Data already extracted, skipping extraction step.")
        # Calculate size mathematically based on bytes to avoid querying DB again
        file_size = MASTER_EMBEDDINGS_PATH.stat().st_size
        total_idx = file_size // (EMBEDDING_DIM * 4)  # float32 = 4 bytes

    # Phase 2: Train Models
    models_exist = all(
        p.is_file() for p in [PCA_MODEL_PATH, UMAP_MODEL_PATH, HDBSCAN_MODEL_PATH]
    )
    if not models_exist:
        train(total_idx)
    else:
        print("Models already trained, skipping training step.")

    # Phase 3: Predict Data
    if not CLUSTERED_DATA_PATH.is_file():
        predict(total_idx)
    else:
        print("Data already clustered and predicted, skipping prediction step.")

    print("\nDataset pipeline execution complete!")


if __name__ == "__main__":
    main()
