import json
import numpy as np
from sentence_transformers import SentenceTransformer, util
from sklearn.metrics.pairwise import cosine_similarity
from collections import defaultdict
import torch

from parameters import ProfileConfig


class MetadataReader:
    """
    Builds a tiny byte-index of the dataset on boot.
    Allows the engine to instantly grab specific lines from the hard drive
    without loading all of the corpus's JSON dictionaries into RAM.
    """

    def __init__(self, filepath):
        self.filepath = filepath
        self.line_offsets = []
        print("Building disk index for metadata...")

        # Read the file purely as bytes to find where each line starts
        with open(filepath, "rb") as f:
            offset = 0
            for line in f:
                self.line_offsets.append(offset)
                offset += len(line)

        # Store as a highly compressed C-array
        self.line_offsets = np.array(self.line_offsets, dtype=np.int64)
        print(f"Index built! Tracking {len(self.line_offsets)} rows.")

    def get_multiple(self, indices):
        """Jumps directly to the required lines on disk without reading the rest of the file."""
        results = []
        with open(self.filepath, "r", encoding="utf-8") as f:
            for idx in indices:
                f.seek(self.line_offsets[idx])
                results.append(json.loads(f.readline()))
        return results


class PoemKeywordExtractor:
    def __init__(
        self, profile: ProfileConfig, embedder: SentenceTransformer | None = None
    ):
        """
        profile  - this instance's dataset config (paths, embedding model/dim,
                   line-scoring settings). See parameters.load_profile().
        embedder - optional, an already-loaded SentenceTransformer to reuse.
                   Pass one in when another profile already loaded the same
                   model, so it isn't loaded into memory twice. If omitted,
                   this profile loads its own.
        """
        self.profile = profile

        self.raw_embeddings = np.memmap(
            profile.master_embeddings_path, dtype=np.float32, mode="r"
        ).reshape(-1, profile.embedding_dim)

        self.cluster_labels = np.load(profile.clustered_data_path)["cluster_labels"]

        # The three processed files must be row-aligned; catch a mismatched or
        # half-copied dataset now rather than as a confusing error mid-request.
        if len(self.cluster_labels) != len(self.raw_embeddings):
            raise ValueError(
                f"Profile '{profile.name}': {len(self.raw_embeddings)} embedding rows "
                f"but {len(self.cluster_labels)} cluster labels -- dataset files don't match."
            )

        self.embedder = (
            embedder
            if embedder is not None
            else SentenceTransformer(profile.embedder_model)
        )

        self.metadata_reader = MetadataReader(profile.metadata_path)

        with open(profile.emotional_anchors_path, "r", encoding="utf-8") as f:
            EMOTIONAL_ANCHORS = json.load(f)

        emotions_dict = EMOTIONAL_ANCHORS.get("emotions", EMOTIONAL_ANCHORS)

        self.line_emotional_embeddings = {}
        self.color_map = {}

        for emotion, details in emotions_dict.items():
            lines = details.get("lines", []) if isinstance(details, dict) else []
            excludes = details.get("excludes", []) if isinstance(details, dict) else []

            if isinstance(details, dict) and "color" in details:
                self.color_map[emotion] = details["color"]

            line_embeddings = self.embedder.encode(lines, convert_to_tensor=True)
            line_centroid = torch.mean(line_embeddings, dim=0)

            if excludes:
                exclude_embeddings = self.embedder.encode(
                    excludes, convert_to_tensor=True
                )
                exclude_centroid = torch.mean(exclude_embeddings, dim=0)

                line_centroid = line_centroid - (
                    profile.line_exclude_attenuation_factor * exclude_centroid
                )

            self.line_emotional_embeddings[emotion] = torch.nn.functional.normalize(
                line_centroid, p=2, dim=0
            )

    def process_poem(
        self,
        poem_embeddings: list[torch.Tensor],
        k: int = 15,
    ) -> tuple[list[int], list[dict]]:
        # 1. Convert the incoming list of lists from JavaScript into a 2D NumPy array
        poem_embeddings_np = np.array(poem_embeddings, dtype=np.float32)

        # 2. Compare the client embeddings against your master dataset
        similarity_matrix = cosine_similarity(poem_embeddings_np, self.raw_embeddings)

        poem_matching_records = []
        valid_clusters_set = set()

        # 3. Iterate directly over the embeddings, NOT the text lines
        for i, embedding in enumerate(poem_embeddings):
            # Get the indices of the top K closest master lines
            top_k_indices = np.argsort(similarity_matrix[i])[-k:][::-1]

            # Look up which clusters those master lines belong to
            neighbor_clusters = self.cluster_labels[top_k_indices]

            # Filter out noise (-1) from the clusters
            valid_neighbor_clusters = [c for c in neighbor_clusters if c != -1]

            # Find emotion of line:.
            keyword_vec = torch.tensor(embedding)

            scores = []

            for emotion, category_vec in self.line_emotional_embeddings.items():
                similarity = util.cos_sim(keyword_vec, category_vec).item()
                scores.append(similarity)

            scores = [
                score * self.profile.line_attenuation_factor
                if score < self.profile.line_emotion_attenuation_threshold
                else score
                for score in scores
            ]

            poem_matching_records.append(
                {
                    "nearest_clusters": valid_neighbor_clusters,
                    "mean_similarity": np.mean(similarity_matrix[i][top_k_indices]),
                    "scores": scores,
                }
            )

            valid_clusters_set.update(valid_neighbor_clusters)

        return list(valid_clusters_set), poem_matching_records

    def select_lines_from_neighborhoods(self, poem_metadata, max_neighborhoods=15):
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
        unique_labels, counts = np.unique(self.cluster_labels, return_counts=True)
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
        valid_mask = np.isin(self.cluster_labels, top_clusters)
        candidate_indices = np.where(valid_mask)[0]

        # Slice everything cleanly while maintaining 100% index alignment
        filtered_metadata = self.metadata_reader.get_multiple(candidate_indices)
        filtered_cluster_labels = self.cluster_labels[candidate_indices]

        return filtered_metadata, filtered_cluster_labels

    def extract_words(
        self,
        poem_metadata,
        filtered_metadata,
        filtered_cluster_labels,
        emotional_weight=0.6,
        semantic_weight=0.4,
        line_weight=0.5,
        word_weight=0.5,
        top_words_to_keep=20,
    ) -> list[dict]:
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

        # --- Normalize Cluster Semantic Profiles (Min-Max Scaling to [0, 1]) ---
        if cluster_semantic_profiles:
            sem_min = min(cluster_semantic_profiles.values())
            sem_max = max(cluster_semantic_profiles.values())
            if sem_max > sem_min:
                cluster_semantic_profiles = {
                    cid: (val - sem_min) / (sem_max - sem_min)
                    for cid, val in cluster_semantic_profiles.items()
                }
            else:
                cluster_semantic_profiles = {
                    cid: 1.0 for cid in cluster_semantic_profiles
                }

        # 3. Rank lines based on their composite scores
        best_keywords = {}

        for i, line_record in enumerate(filtered_metadata):
            cluster_id = filtered_cluster_labels[i]

            line_scores = line_record.get("score")
            keywords = line_record.get("keywords")

            if line_scores is None or cluster_id not in cluster_emotional_profiles:
                continue

            line_scores_arr = np.array(line_scores)

            # --- Normalize Line Emotional Scores (Min-Max Scaling to [0, 1]) ---
            line_min_val = np.min(line_scores_arr)
            line_max_val = np.max(line_scores_arr)
            if line_max_val > line_min_val:
                line_scores_arr = (line_scores_arr - line_min_val) / (
                    line_max_val - line_min_val
                )
            else:
                line_scores_arr = np.zeros_like(line_scores_arr)

            cluster_profile_arr = cluster_emotional_profiles[cluster_id]

            # --- Line Emotional Closeness ---
            emo_distance = np.linalg.norm(line_scores_arr - cluster_profile_arr)
            line_emotional_closeness = 1.0 / (1.0 + emo_distance)

            # --- Cluster Semantics Closeness ---
            cluster_semantic_closeness = cluster_semantic_profiles.get(cluster_id, 0.0)

            # --- WORD SCORE CALCULATION  ---

            for keyword in keywords:
                word_scores = keyword.get("score")
                word_scores_arr = np.array(word_scores)

                # --- Normalize Word Emotional Scores (Min-Max Scaling to [0, 1]) ---
                word_min_val = np.min(word_scores_arr)
                word_max_val = np.max(word_scores_arr)
                if word_max_val > word_min_val:
                    word_scores_arr = (word_scores_arr - word_min_val) / (
                        word_max_val - word_min_val
                    )
                else:
                    word_scores_arr = np.zeros_like(word_scores_arr)

                # --- Word Emotional Closeness ---
                word_emo_distance = np.linalg.norm(
                    word_scores_arr - cluster_profile_arr
                )
                word_emotional_closeness = 1.0 / (1.0 + word_emo_distance)

                # 1. Combine them using a weighted average
                combined_emotional_closeness = (
                    word_weight * word_emotional_closeness
                ) + (line_weight * line_emotional_closeness)

                # 2. Apply global weights
                semantic_sim_score = semantic_weight * cluster_semantic_closeness
                emotional_sim_score = emotional_weight * combined_emotional_closeness

                final_score = semantic_sim_score + emotional_sim_score
                word_text = keyword["word"]

                if (
                    word_text not in best_keywords
                    or final_score > best_keywords[word_text]["score"]
                ):
                    best_keywords[word_text] = {
                        "word": word_text,
                        "color": keyword["color"],
                        "score": final_score,
                        "inverseFrequency": keyword["inverseFrequency"],
                    }

        ranked_keywords = list(best_keywords.values())
        ranked_keywords.sort(key=lambda x: x["score"], reverse=True)
        final_keywords = ranked_keywords[:top_words_to_keep]

        return final_keywords

    def display_extracted_words(self, final_keywords):
        for keyword in final_keywords:
            print(
                f"word: {keyword['word']} | color: {keyword['color']} | score: {keyword['score']:.4f}"
            )

    def extract(
        self,
        poem_embeddings,
        weights,
    ):
        # --- 1. Process Input Poem, Generate Neighborhoods ---
        poem_clusters_set, poem_metadata = self.process_poem(
            poem_embeddings, k=weights.context_line_amount
        )

        if not poem_clusters_set:
            print("No neighborhoods found for this poem.")
            return
        else:
            print(f"{len(poem_clusters_set)} neighborhoods found for this poem.")

        # --- 2. Extract All Lines from K Best Neighborhoods ---
        filtered_metadata, filtered_cluster_labels = (
            self.select_lines_from_neighborhoods(
                poem_metadata, max_neighborhoods=weights.neighborhood_amount
            )
        )

        # --- 3. Extract Words from Lines ---
        neighborhood_words = self.extract_words(
            poem_metadata,
            filtered_metadata,
            filtered_cluster_labels,
            emotional_weight=weights.emotional_weight,
            semantic_weight=1 - weights.emotional_weight,
            line_weight=weights.line_weight,
            word_weight=1 - weights.line_weight,
            top_words_to_keep=weights.max_results,
        )

        return neighborhood_words
