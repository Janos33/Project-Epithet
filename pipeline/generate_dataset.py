import os

from neo4j import GraphDatabase, Driver
from sklearn.decomposition import PCA
from dotenv import load_dotenv
import hdbscan
import json
import numpy
import umap
import gc

RAW_DATA_DIR = "data/raw"
SEMI_PROCESSED_DIR = "data/semi-processed"
PROCESSED_DIR = "data/processed"

EXTRACTED_COORDS_PATH = os.path.join(SEMI_PROCESSED_DIR, "coords.npz")
METADATA_PATH = os.path.join(PROCESSED_DIR, "metadata.json")
CLUSTERED_DATA_PATH = os.path.join(PROCESSED_DIR, "clustered_data.npz")
MEMMAP_PATH = os.path.join(SEMI_PROCESSED_DIR, "raw_embeddings.dat")

def get_neo4j_driver() -> Driver:
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USER = os.getenv("NEO4J_USER")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def extract_data():

    all_metadata = []

    print("Connecting to driver")
    with get_neo4j_driver() as driver:

        batch_size = 50000
        skip = 0
        current_idx = 0

        embedding_dim = 384

        with driver.session() as session:
            total_count = session.execute_read(lambda tx: tx.run("MATCH (n:Line) RETURN count(n) AS cnt").single()["cnt"])

        raw_embeddings = numpy.memmap(MEMMAP_PATH, dtype='float32', mode='w+', shape=(total_count, embedding_dim))


        while True:
            with driver.session() as session:
                records = session.execute_read(
                    lambda tx: tx.run(
                    """
                    MATCH (n:Line)
                    WITH n
                    ORDER BY elementId(n)
                    SKIP $skip LIMIT $batch_size
                    OPTIONAL MATCH (n)-[:HAS_KEYWORD]->(k)
                    RETURN n.text AS lineText, 
                        n.embedding AS lineEmbedding, 
                        collect({word: k.word, color: k.color, emotion: k.emotion, score: k.score}) AS keywords
                        """,
                    skip=skip,
                    batch_size=batch_size,
                ).data()
            )

            # Exit the loop if no more records are returned
            if not records:
                break
        
            for record in records:

                raw_embeddings[current_idx] = record["lineEmbedding"]
                lineData = {"lineText": record["lineText"], "keywords": record["keywords"]}
                all_metadata.append(lineData)

                current_idx += 1

            print(
                f"Processed batch starting at index {skip} (Total in batch:"
                f" {len(records)})"
            )
            skip += batch_size

    raw_embeddings.flush()

    # In case of keywordless lines, trim unused memory slots before PCA.
    valid_embeddings = raw_embeddings[:current_idx]
    
    print("Running PCA to reduce high-dimensional space...")
    pca = PCA(n_components=50, random_state=42)
    pca_embeddings = pca.fit_transform(valid_embeddings)

    print("Running UMAP dimensionality reduction...")
    reducer = umap.UMAP(n_components=10, n_neighbors=15, min_dist=0.1, metric='cosine', init='random', verbose=True, random_state=42)
    umap_coords = reducer.fit_transform(pca_embeddings)

    print(f"Saving reduced coordinates to {EXTRACTED_COORDS_PATH}")
    print(f"Saving metadata to {METADATA_PATH}")
    numpy.savez_compressed(EXTRACTED_COORDS_PATH, coords=umap_coords)
    with open(METADATA_PATH, "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)

    # Clean up all memory-map references
    if "valid_embeddings" in locals():
        del valid_embeddings
    del raw_embeddings

    # Force garbage collection to release the file lock on Windows
    gc.collect()

    # Safely remove the temp file
    if os.path.exists(MEMMAP_PATH):
        os.remove(MEMMAP_PATH)
    print("Dataset generation complete!")

    print(f"Successfully processed and saved {len(all_metadata)} lines/")

def cluster_data():

    print("Loading semi-processed coordinates...")
    with numpy.load(EXTRACTED_COORDS_PATH) as data:

        if "coords" in data:
            coords = data["coords"]
        else:
            coords = data[data.files[0]]

    print("Running HDBSCAN clustering on the dataset...")
    clusterer = hdbscan.HDBSCAN(min_cluster_size=15, min_samples=5, metric="euclidean")
    cluster_labels = clusterer.fit_predict(coords)

    print(f"Clustering complete. Found {len(set(cluster_labels)) - (1 if -1 in cluster_labels else 0)} valid clusters.")

    print(f"Saving clustered data to {CLUSTERED_DATA_PATH}...")
    numpy.savez_compressed(
        CLUSTERED_DATA_PATH,
        coords=coords,
        cluster_labels=cluster_labels
    )
    print("Clustered data successfully saved.")

def main():

    os.makedirs(RAW_DATA_DIR, exist_ok=True)
    os.makedirs(SEMI_PROCESSED_DIR, exist_ok=True)
    os.makedirs(PROCESSED_DIR, exist_ok=True)

    print("Data processing started")
    
    load_dotenv()

    if not os.path.isfile(EXTRACTED_COORDS_PATH) or not os.path.isfile(METADATA_PATH):
        print("Start of data extraction")
        extract_data()
    else:
        print("Data already has been extracted, skipping step")

    if not os.path.isfile(CLUSTERED_DATA_PATH):
        print("Start of data clustering")
        cluster_data()
    else:
        print("Data already has been clustered, skipping step")

    print("Data processing over")

if __name__ == "__main__":
    main()