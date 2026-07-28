import os

from neo4j import GraphDatabase, Driver
from sklearn.decomposition import PCA
from dotenv import load_dotenv
import json
import numpy
import umap

def get_neo4j_driver() -> Driver:
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USER = os.getenv("NEO4J_USER")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def extract_data():

    all_metadata = []

    print("Connecting to driver")
    with get_neo4j_driver() as driver:

        output_dir = "data/processed"
        os.makedirs(output_dir, exist_ok=True)

        batch_size = 50000
        skip = 0
        current_idx = 0

        embedding_dim = 384

        with driver.session() as session:
            total_count = session.execute_read(lambda tx: tx.run("MATCH (n:Line) RETURN count(n) AS cnt").single()["cnt"])

        memmap_path = os.path.join(output_dir, "raw_embeddings.dat")
        raw_embeddings = numpy.memmap(memmap_path, dtype='float32', mode='w+', shape=(total_count, embedding_dim))


        while True:
            with driver.session() as session:
                records = session.execute_read(
                    lambda tx: tx.run(
                    """
                    MATCH (n:Line)
                    WITH n
                    SKIP $skip LIMIT $batch_size
                    MATCH (n)-[:HAS_KEYWORD]->(k)
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

    print("Running PCA to reduce high-dimensional space...")
    pca = PCA(n_components=50, random_state=42)
    pca_embeddings = pca.fit_transform(raw_embeddings)

    print("Running UMAP dimensionality reduction...")
    reducer = umap.UMAP(n_components=10, n_neighbors=15, min_dist=0.1, metric='cosine', init='random', verbose=True, random_state=42)
    umap_coords = reducer.fit_transform(pca_embeddings)

    # Save final reduced coordinates and metadata
    numpy.savez_compressed(os.path.join(output_dir, "coords.npz"), coords=umap_coords)
    with open(os.path.join(output_dir, "metadata.json"), "w", encoding="utf-8") as f:
        json.dump(all_metadata, f, ensure_ascii=False, indent=2)
        
    # Clean up the temporary dat file
    del raw_embeddings
    os.remove(memmap_path)
    print("Dataset generation complete!")

    print(f"Successfully processed and saved {len(all_metadata)} lines to {output_dir}/")

def main():
    load_dotenv()
    extract_data()

if __name__ == "__main__":
    main()