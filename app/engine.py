import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
from parameters import *
import torch


def load_dataset():
    raw_embeddings = np.load(MASTER_EMBEDDINGS_PATH)

    clustered_npz = np.load(COORDS_PATH)

    cluster_labels = clustered_npz["cluster_labels"]

    with open(METADATA_PATH, "r", encoding="utf-8") as f:
        metadata = json.load(f)

    return raw_embeddings, cluster_labels, metadata


def process_poem(
    poem_text: str,
    master_embeddings: np.ndarray,
    cluster_labels: np.ndarray,
    k: int = 15,
) -> tuple[list[int], pd.DataFrame]:
    poem_split = [line.strip() for line in poem_text.splitlines() if line.strip()]

    # Calculate emotional embeddings

    embedder = SentenceTransformer("all-MiniLM-L6-v2")
    EMOTIONAL_EMBEDDINGS = {}

    for emotion, words in EMOTIONAL_ANCHORS.items():
        # Encode all words for an emotion, then average them into a single vector (centroid)
        word_embeddings = embedder.encode(words, convert_to_tensor=True)
        EMOTIONAL_EMBEDDINGS[emotion] = torch.mean(word_embeddings, dim=0)

    poem_embeddings = [embedder.encode(line) for line in poem_split]

    similarity_matrix = cosine_similarity(poem_embeddings, master_embeddings)

    poem_matching_records = []
    valid_clusters_set = set()

    for i, line in enumerate(poem_split):
        # Get the indices of the top K closest master lines
        top_k_indices = np.argsort(similarity_matrix[i])[-k:][::-1]

        # Look up which clusters those master lines belong to
        neighbor_clusters = cluster_labels[top_k_indices]

        # Filter out noise (-1) from the clusters
        valid_neighbor_clusters = [c for c in neighbor_clusters if c != -1]

        # Find emotion of line
        keyword_vec = embedder.encode(line, convert_to_tensor=True)

        scores = []

        for emotion, category_vec in EMOTIONAL_EMBEDDINGS.items():
            similarity = util.cos_sim(keyword_vec, category_vec).item()
            scores.append(similarity)

        scores = [
            score * ATTENUATION_FACTOR if score < LINE_EMOTION_THRESHOLD else score
            for score in scores
        ]

        poem_matching_records.append(
            {
                "lineText": line,
                "nearest_clusters": valid_neighbor_clusters,
                "mean_similarity": np.mean(similarity_matrix[i][top_k_indices]),
                "scores": scores,
            }
        )

        valid_clusters_set.update(valid_neighbor_clusters)

    return list(valid_clusters_set), poem_matching_records


def select_lines_from_neighborhoods(
    poem_metadata, metadata, cluster_labels, max_neighborhoods=15
):
    """
    Step 3: Track, rank, filter neighborhoods, and extract matching master rows vectorially.
    """
    # 1. TRACKING: Aggregate hits and similarity scores for each cluster
    cluster_stats = defaultdict(lambda: {"hit_count": 0, "total_similarity": 0.0})

    for record in poem_metadata:
        sim_score = record.get("mean_similarity", 0.0)

        for cluster_id in record["nearest_clusters"]:
            if cluster_id == -1:
                continue

            cluster_stats[cluster_id]["hit_count"] += 1
            cluster_stats[cluster_id]["total_similarity"] += sim_score

    # 2. RANKING: Compute a combined score (Volume × Confidence)

    unique_labels, counts = np.unique(cluster_labels, return_counts=True)
    global_cluster_sizes = dict(zip(unique_labels, counts))

    ranked_clusters = []
    for cluster_id, stats in cluster_stats.items():
        avg_sim = stats["total_similarity"] / stats["hit_count"]
        hit_weight = np.log2(stats["hit_count"] + 1)
        size_penalty = 1.0 / (
            1.0 + np.exp((global_cluster_sizes[cluster_id] - 5000) / 2500)
        )

        score = hit_weight * avg_sim * size_penalty

        ranked_clusters.append((cluster_id, score))

    # Sort descending by score (best clusters first)
    ranked_clusters.sort(key=lambda x: x[1], reverse=True)

    # 3. FILTERING: Keep only the top K neighborhoods (e.g., max 15)
    top_clusters = [
        cluster_id for cluster_id, score in ranked_clusters[:max_neighborhoods]
    ]

    print(
        f"Filtered down to top {len(top_clusters)} neighborhoods (from {len(cluster_stats)} raw matches)."
    )

    # 4. VECTORIZED EXTRACTION: Pull all master lines belonging to these top clusters instantly
    valid_mask = np.isin(cluster_labels, top_clusters)
    candidate_indices = np.where(valid_mask)[0]

    # Slice everything cleanly while maintaining 100% index alignment
    filtered_metadata = [metadata[i] for i in candidate_indices]
    filtered_cluster_labels = cluster_labels[candidate_indices]

    return filtered_metadata, filtered_cluster_labels


def extract_words(
    poem_metadata,
    filtered_metadata,
    filtered_cluster_labels,
    emotional_weight=0.6,
    semantic_weight=0.4,
) -> dict[int, list[dict]]:
    # 1. Collect score vectors and mean similarities for each cluster from the input poem
    cluster_score_lists = defaultdict(list)
    cluster_similarity_lists = defaultdict(list)

    for record in poem_metadata:
        scores = record.get("scores")
        mean_sim = record.get("mean_similarity", 0.5)
        if scores is None:
            continue

        for cluster_id in record["nearest_clusters"]:
            if cluster_id == -1:
                continue
            cluster_score_lists[cluster_id].append(scores)
            cluster_similarity_lists[cluster_id].append(mean_sim)

    # 2. Compute mathematical means for each cluster's profile and semantic baseline
    cluster_emotional_profiles = {}
    for cluster_id, score_list in cluster_score_lists.items():
        cluster_emotional_profiles[cluster_id] = np.mean(score_list, axis=0)

    cluster_semantic_profiles = {}
    for cluster_id, sim_list in cluster_similarity_lists.items():
        cluster_semantic_profiles[cluster_id] = np.mean(sim_list)

    # 3. Rank lines based on their composite scores
    best_keywords = {}

    for i, line_record in enumerate(filtered_metadata):
        cluster_id = filtered_cluster_labels[i]

        line_scores = line_record.get("score")
        keywords = line_record.get("keywords")

        if line_scores is None or cluster_id not in cluster_emotional_profiles:
            continue

        line_scores_arr = np.array(line_scores)
        cluster_profile_arr = cluster_emotional_profiles[cluster_id]

        # --- Line Emotional Closeness ---
        emo_distance = np.linalg.norm(line_scores_arr - cluster_profile_arr)
        line_emotional_closeness = 1.0 / (1.0 + emo_distance)

        # --- Cluster Semantics Closeness ---
        cluster_semantic_closeness = cluster_semantic_profiles.get(cluster_id)

        # --- WORD SCORE CALCULATION  ---

        for keyword in keywords:
            word_scores = keyword.get("score")
            word_scores_arr = np.array(word_scores)

            # --- Word Emotional Closeness ---
            word_emo_distance = np.linalg.norm(word_scores_arr - cluster_profile_arr)
            word_emotional_closeness = 1.0 / (1.0 + word_emo_distance)

            semantic_sim_score = semantic_weight * cluster_semantic_closeness
            line_sim_score = emotional_weight * line_emotional_closeness
            word_sim_score = emotional_weight * word_emotional_closeness

            final_score = semantic_sim_score + line_sim_score * word_sim_score

            word_text = keyword["word"]

            if (
                word_text not in best_keywords
                or final_score > best_keywords[word_text]["score"]
            ):
                best_keywords[word_text] = {
                    "word": word_text,
                    "color": keyword["color"],
                    "score": final_score,
                }

    ranked_keywords = list(best_keywords.values())

    ranked_keywords.sort(key=lambda x: x["score"], reverse=True)
    final_keywords = ranked_keywords[:TOP_WORDS_TO_KEEP]

    return final_keywords


def display_extracted_words(final_keywords):
    for keyword in final_keywords:
        print(
            f"word: {keyword['word']} | color: {keyword['color']} | line_score: {keyword['line_score']}"
        )


def find_keywords(
    poem_text, emotional_weight, raw_embeddings, cluster_labels, metadata
):
    # --- 1. Process Input Poem, Generate Neighborhoods ---
    # Poem_clusters_set - the list of clusters that are closest to at least 1 line
    # Poem_metadata - poem lines, their nearest clusters, how close said clusters are, emotional scores

    poem_clusters_set, poem_metadata = process_poem(
        poem_text, raw_embeddings, cluster_labels, K_LINES_TO_GET
    )

    if not poem_clusters_set:
        print("No neighborhoods found for this poem.")
        return
    else:
        print(f"{len(poem_clusters_set)} neighborhoods found for this poem.")

    # --- 2. Extract All Lines from K Best Neighborhoods ---
    # Filtered versions of each variable that only contain the data of relevant neighborhoods

    filtered_metadata, filtered_cluster_labels = select_lines_from_neighborhoods(
        poem_metadata, metadata, cluster_labels, K_NEIGHBORHOODS_TO_KEEP
    )

    # --- 3. Extract Words from Lines ---

    neighborhood_words = extract_words(
        poem_metadata,
        filtered_metadata,
        filtered_cluster_labels,
        emotional_weight,
        1 - emotional_weight,
    )

    return neighborhood_words
