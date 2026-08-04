from collections import Counter
import json
import math
import numpy
import pandas
from sentence_transformers import SentenceTransformer
from sklearn.decomposition import PCA
from sklearn.neighbors import NearestNeighbors
import umap
import hdbscan
from parameters import *

#--- 1. Setup: Load Base Dataset ---

clustered_npz = numpy.load(COORDS_PATH)
base_coords = clustered_npz["coords"]
cluster_labels = clustered_npz["cluster_labels"]

with open(METADATA_PATH, "r", encoding="utf-8") as f:
    metadata = json.load(f)

# Bind base dataset to a DataFrame for easy querying
df = pandas.DataFrame(metadata)
df["cluster"] = cluster_labels


# --- 2. Process Input Poem ---
poem_split = [line.strip() for line in POEM_TEXT.splitlines() if line.strip()]

embedder = SentenceTransformer("all-MiniLM-L6-v2")
poem_embeddings = [embedder.encode(line) for line in poem_split]

pca = PCA(n_components=PCA_parameters.n_components, random_state=PCA_parameters.random_state)
pca_embeddings = pca.fit_transform(poem_embeddings)

reducer = umap.UMAP(
    n_components=UMAP_parameters.n_components,
    n_neighbors=UMAP_parameters.n_neighbors,
    min_dist=UMAP_parameters.min_dist,
    metric=UMAP_parameters.metric,
    random_state=UMAP_parameters.random_state
)
poem_umap_coords = reducer.fit_transform(pca_embeddings)


# --- 3. Rule 1: Find Neighborhoods via HDBSCAN & Filter ---
def get_valid_neighborhoods(coords, min_cluster_size=3):
    total_lines = len(coords)
    target_threshold = MINIMUM_LARGEST_NEIGHBORHOOD_SIZE * total_lines

    current_min_size = min_cluster_size
    clusterer = None
    labels = []

    # Dynamically increase size/loosen constraints if neighborhoods are too small
    while current_min_size <= max(2, total_lines // 2):
        clusterer = hdbscan.HDBSCAN(
            min_cluster_size=current_min_size, min_samples=1, metric="euclidean"
        )
        labels = clusterer.fit_predict(coords)

        # Count sizes of each valid cluster (ignoring noise '-1')
        counts = Counter(labels)
        if -1 in counts:
            del counts[-1]

        if any(size >= target_threshold for size in counts.values()):
            break

        current_min_size += 1

    return labels, clusterer


poem_labels, _ = get_valid_neighborhoods(poem_umap_coords)
poem_df = pandas.DataFrame({"line": poem_split, "cluster": poem_labels})

# Filter out neighborhoods that don't meet the size constraint
valid_clusters = [
    cluster_id
    for cluster_id, count in Counter(poem_labels).items()
    if cluster_id != -1 and count >= (MINIMUM_NEIGHBORHOOD_SIZE * len(poem_split))
]


# --- 4. Rule 2: Check Remaining Lines (<95% similar) up to target count ---
num_neighborhoods = max(len(valid_clusters), 1)
target_line_quota = int(TARGET_LINE_QUOTA_WHOLE / num_neighborhoods)

# Use NearestNeighbors to check similarity against base dataset
nn = NearestNeighbors(n_neighbors=NearestNeighbors_parameters.n_neighbors, metric=NearestNeighbors_parameters.metric).fit(base_coords)
distances, indices = nn.kneighbors(poem_umap_coords)

extracted_neighborhood_data = {}

for cluster_id in valid_clusters:
    cluster_lines_df = poem_df[poem_df["cluster"] == cluster_id]
    
    collected_items = []
    collected_texts = set()

    # 1. Add initial poem lines from this cluster as full dictionaries (Similarity = 1.0)
    for _, row in cluster_lines_df.iterrows():
        item_dict = row.to_dict()
        line_text = item_dict.get("lineText") or item_dict.get("line")
        
        if line_text and line_text not in collected_texts:
            item_dict["line_similarity"] = 1.0
            collected_texts.add(line_text)
            collected_items.append(item_dict)

    # 2. Find remaining lines in base dataset matching similarity criteria
    for idx_in_poem, _ in cluster_lines_df.iterrows():
        if len(collected_items) >= target_line_quota:
            break

        neighbor_idxs = indices[idx_in_poem]
        neighbor_dists = distances[idx_in_poem]

        for n_idx, dist in zip(neighbor_idxs, neighbor_dists):
            similarity = 1.0 - dist
            
            # Check <95% similarity condition to avoid near-identical duplicates
            if similarity < MAX_LINE_SIMILARITY_THRESHOLD:
                neighbor_item = df.iloc[n_idx].to_dict()
                neighbor_item["line_similarity"] = float(similarity)
                matching_text = neighbor_item["lineText"]

                if matching_text not in collected_texts:
                    collected_texts.add(matching_text)
                    collected_items.append(neighbor_item)

                if len(collected_items) >= target_line_quota:
                    break

    extracted_neighborhood_data[cluster_id] = collected_items


# --- 5. Rule 3: Extract Emotionally Strongest Words (Score * Line Similarity) ---
seen_words = set()

num_neighborhoods = len(extracted_neighborhood_data)
word_quota = int(WORD_QUOTA_WHOLE / num_neighborhoods) if num_neighborhoods > 0 else WORD_QUOTA_WHOLE

neighborhood_words = {}

for cluster_id, items in extracted_neighborhood_data.items():
    all_keywords = []
    
    # Collect all keyword dicts and attach line_similarity + final_score
    for item in items:
        if isinstance(item, dict):
            keywords = item.get("keywords", [])
            line_sim = item.get("line_similarity", 1.0)
            
            for kw in keywords:
                kw_entry = dict(kw)
                kw_entry["line_similarity"] = line_sim
                kw_entry["final_score"] = min(FINAL_SCORE_MAX, kw_entry["score"]) * (1.0 + 0.1 * math.log(1 / (kw_entry["frequency"] + 1e-6))) * (1+line_sim ** 12)
                all_keywords.append(kw_entry)

    # Sort keywords by composite score (score * line_similarity)
    sorted_keywords = sorted(all_keywords, key=lambda k: k["final_score"], reverse=True)

    # Keep only the highest composite-scoring instance of each word
    unique_keywords = []
    for kw in sorted_keywords:
        word_key = kw["word"].lower()
        if word_key not in seen_words:
            seen_words.add(word_key)
            unique_keywords.append(kw)

    neighborhood_words[cluster_id] = unique_keywords[:word_quota]

print("Neighborhood Extraction Complete!")
print(f"Valid Neighborhoods Found: {num_neighborhoods}")


# --- 6. Print All Extracted Words ---
print("\n" + "=" * 80)
print("EXTRACTED WORDS (SORTED BY SCORE * LINE SIMILARITY)")
print("=" * 80)

for cluster_id, keywords in neighborhood_words.items():
    print(f"\n[ Cluster / Neighborhood ID: {cluster_id} ]")
    if not keywords:
        print("  (No keywords found)")
        continue
        
    for kw in keywords:
        word = kw["word"]
        color = kw["color"]
        emotion = kw.get("emotion", "N/A")
        raw_score = kw.get("score", 0.0)
        line_sim = kw.get("line_similarity", 1.0)
        final_score = kw.get("final_score", 0.0)
        
        print(
            f"Word: {word:<15} | Color: {color} | Emotion: {emotion:<12} | "
            f"Raw: {raw_score:.3f} | Sim: {line_sim:.3f} | Frequency: {kw['frequency']:<5} | Final: {final_score:.3f}"
        )