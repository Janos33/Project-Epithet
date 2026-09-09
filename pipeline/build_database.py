"""
Poetry Graph Ingestion Pipeline
-------------------------------
This script reads a JSON corpus of poetry, extracts emotional keywords using
NLP (spaCy) and SentenceTransformers, and ingests the data into a Neo4j graph database.
"""

import datetime
import hashlib
import json
import os
import re
import neo4j
import numpy as np
from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver
from sentence_transformers import SentenceTransformer, util
from functools import lru_cache
import spacy
import torch
from parameters import *


def get_neo4j_driver() -> Driver:
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USER = os.getenv("NEO4J_USER")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


class PoetryGraphPipeline:
    def __init__(self):
        print("Loading models and emotional anchors...")
        self.embedder = SentenceTransformer(EMBEDDER_MODEL)
        self.nlp = spacy.load(SPACY_MODEL, disable=["parser", "ner"])

        # Pre-compute the emotional vectors so we aren't doing it per-poem
        anchors = self._process_anchors()
        self.word_emotional_embeddings = anchors[0]
        self.line_emotional_embeddings = anchors[1]
        self.color_map = anchors[2]

        self.stacked_word_emotions = torch.stack(
            list(self.word_emotional_embeddings.values())
        )
        self.stacked_line_emotions = torch.stack(
            list(self.line_emotional_embeddings.values())
        )
        self.emotion_keys = list(self.line_emotional_embeddings.keys())

        # Pre-parse hex strings to RGB tuples for faster color blending later
        self.color_rgb_map = {}
        for emotion, hex_str in self.color_map.items():
            h = hex_str.lstrip("#")
            self.color_rgb_map[emotion] = (
                int(h[0:2], 16),
                int(h[2:4], 16),
                int(h[4:6], 16),
            )

    def _process_anchors(self) -> tuple[dict, dict, dict]:
        with open(EMOTIONAL_ANCHORS_PATH, "r", encoding="utf-8") as f:
            emotional_anchors_data = json.load(f)

        word_emotional_embeddings = {}
        line_emotional_embeddings = {}
        color_map = {}

        emotions = emotional_anchors_data.get("emotions", emotional_anchors_data)

        # Build a "centroid" vector for each emotion based on its anchor phrases.
        # If there are exclusion words, subtract their meaning to sharpen the anchor.
        for emotion, details in emotions.items():
            lines = details["lines"]
            words = details["words"]
            excludes = details.get("excludes", [])

            if "color" in details:
                color_map[emotion] = details["color"]

            word_embeddings = self.embedder.encode(words, convert_to_tensor=True)
            word_centroid = torch.mean(word_embeddings, dim=0)

            line_embeddings = self.embedder.encode(lines, convert_to_tensor=True)
            line_centroid = torch.mean(line_embeddings, dim=0)

            if excludes:
                exclude_embeddings = self.embedder.encode(
                    excludes, convert_to_tensor=True
                )
                exclude_centroid = torch.mean(exclude_embeddings, dim=0)

                word_centroid = word_centroid - (0.5 * exclude_centroid)
                line_centroid = line_centroid - (0.5 * exclude_centroid)

            word_emotional_embeddings[emotion] = torch.nn.functional.normalize(
                word_centroid, p=2, dim=0
            )
            line_emotional_embeddings[emotion] = torch.nn.functional.normalize(
                line_centroid, p=2, dim=0
            )

        return word_emotional_embeddings, line_emotional_embeddings, color_map

    def batch_ingest_lines(
        self,
        line_batch: list[dict],
        session: neo4j.Session,
        counter: int,
        total_lines_ingested: int,
    ):
        # Uses MERGE instead of CREATE so the script is idempotent (safe to restart if it crashes)
        query = """
        UNWIND $batch AS row

        MERGE (w:Author {name: row.author})

        MERGE (l:Line {id: row.id})
        ON CREATE SET
            l.text = row.line_text,
            l.embedding = row.embedding,
            l.score = row.line_score,
            l.createdAt = datetime(),
            l.updatedAt = datetime()
        ON MATCH SET
            l.text = row.line_text,
            l.embedding = row.embedding,
            l.score = row.line_score,
            l.updatedAt = datetime()

        MERGE (w)-[:AUTHORED]->(l)

        WITH l, row
        WHERE row.keywords IS NOT NULL AND size(row.keywords) > 0
        UNWIND range(0, size(row.keywords) - 1) AS i

        WITH l, 
            row.keywords[i] AS kw_word, 
            row.keyword_score[i] AS kw_score, 
            row.keyword_color[i] AS kw_color,
            i AS kw_index

        MERGE (k:PoemKeyword {text: kw_word})

        MERGE (l)-[r:HAS_KEYWORD {position: kw_index}]->(k)  
        ON CREATE SET 
            r.score = kw_score,
            r.color = kw_color,
            r.createdAt = datetime()
        ON MATCH SET
            r.score = kw_score,
            r.color = kw_color,
            r.updatedAt = datetime()
        """

        session.run(query, batch=line_batch)
        print(
            f"[{datetime.datetime.now()}] Ingested batch #{counter} into Neo4j - batch size: {len(line_batch)} - ingested lines: {total_lines_ingested}"
        )

    @lru_cache(maxsize=LRU_CACHE_SIZE)
    def classify_raw_word_emotional_tone(self, text: str) -> list:
        # Cache this so we aren't re-running the transformer for common words like "love" or "dark"
        keyword_vec = self.embedder.encode(text, convert_to_tensor=True)
        similarities = util.cos_sim(keyword_vec, self.stacked_word_emotions)[0]
        return similarities.tolist()

    def classify_word_emotional_tone(
        self, text: str, line_scores: list, alpha: float = 0.2
    ) -> list:
        # Tweak the word's inherent emotion using the context of the line it sits in
        raw_scores = self.classify_raw_word_emotional_tone(text)
        scores = []

        for idx, similarity in enumerate(raw_scores):
            adjusted_score = (similarity * (1.0 - alpha)) + (line_scores[idx] * alpha)
            scores.append(adjusted_score)

        scores = [
            score * ATTENUATION_FACTOR if score < WORD_EMOTION_THRESHOLD else score
            for score in scores
        ]

        return scores

    def extract_keywords(self, doc: "spacy.tokens.Doc", line_scores: list) -> list:
        keywords = []

        # Grab meaningful parts of speech, skip stop words like "the" or "and"
        for token in doc:
            if (
                token.pos_ in {"NOUN", "ADJ", "VERB", "ADV"}
                and not token.is_stop
                and token.is_alpha
            ):
                keywords.append(token.lemma_.lower())

        keyword_classifications = []
        for word_str in keywords:
            word_score = self.classify_word_emotional_tone(word_str, line_scores)

            if max(word_score) > WORD_EMOTION_THRESHOLD:
                color = self.get_blended_color(word_score)
                keyword_classifications.append(
                    {"word": word_str, "score": word_score, "color": color}
                )

        return keyword_classifications

    def get_blended_color(
        self, scores: list, top_k: int = 4, power: float = 4.0
    ) -> str:
        # Takes the top scoring emotions and mixes their hex colors based on their relative strength
        scores = np.array(scores, dtype=float)
        scores = np.maximum(scores, 0.0)

        # Force Neutral to drop out so it doesn't muddy the vibrant colors
        if "Neutral" in self.emotion_keys:
            neutral_idx = self.emotion_keys.index("Neutral")
            scores[neutral_idx] = 0.0

        top_k = min(top_k, np.count_nonzero(scores))
        if top_k == 0:
            return self.color_map.get("Neutral", "#A0A0A0")

        top_indices = np.argsort(scores)[-top_k:][::-1]
        top_scores = scores[top_indices]
        max_score = top_scores[0]

        if max_score <= 0:
            return self.color_map.get("Neutral", "#A0A0A0")

        # Exaggerate the top score differences so the strongest emotion dictates the primary hue
        relative_ratios = (top_scores / max_score) ** power
        weights = relative_ratios / relative_ratios.sum()

        r_sq_sum, g_sq_sum, b_sq_sum = 0.0, 0.0, 0.0

        for idx, weight in zip(top_indices, weights):
            emotion = self.emotion_keys[idx]
            if emotion in self.color_rgb_map:
                r, g, b = self.color_rgb_map[emotion]
                r_sq_sum += weight * (r**2)
                g_sq_sum += weight * (g**2)
                b_sq_sum += weight * (b**2)

        final_r = np.sqrt(r_sq_sum)
        final_g = np.sqrt(g_sq_sum)
        final_b = np.sqrt(b_sq_sum)

        return "#{:02X}{:02X}{:02X}".format(
            int(round(final_r)), int(round(final_g)), int(round(final_b))
        )

    def ingest_data(self, session: neo4j.Session):
        current_batch = []
        counter = 0
        total_lines_ingested = 0

        print("Loading dataset...")
        with open(RAW_POEMS_PATH, "r", encoding="utf-8") as f:
            des_data = json.load(f)

        for poem in des_data:
            author = poem.get("Author", "Unknown")
            title = poem.get("Title", "Untitled")
            text = poem.get("text", "")
            poem_hash = hashlib.md5(text.encode("utf-8")).hexdigest()[:8]

            # Handle edge cases: split super long lines so we don't blow up the transformer context limits
            lines_to_process = title + "\n" + text
            current_combined = ""
            processed_lines = []

            for raw_line in lines_to_process.splitlines():
                raw_line = raw_line.strip()
                if not raw_line:
                    continue

                if len(raw_line) > MAX_LINE_LENGTH:
                    parts = [
                        s.strip()
                        for s in re.split(r"(?<=[,;:!?.])\s+", raw_line)
                        if s.strip()
                    ]
                    sub_lines = []
                    for p in parts:
                        while len(p) > MAX_LINE_LENGTH:
                            split_idx = p.rfind(" ", 0, MAX_LINE_LENGTH)
                            if split_idx == -1:
                                split_idx = MAX_LINE_LENGTH
                            sub_lines.append(p[:split_idx].strip())
                            p = p[split_idx:].strip()
                        if p:
                            sub_lines.append(p)
                else:
                    sub_lines = [raw_line]

                # Recombine tiny lines together to give the model more context
                for line in sub_lines:
                    current_combined = (
                        (current_combined + " " + line) if current_combined else line
                    )

                    if len(current_combined) >= LINE_LENGTH_THRESHOLD:
                        processed_lines.append(current_combined)
                        current_combined = ""

            if current_combined:
                if processed_lines and len(current_combined) < LINE_LENGTH_THRESHOLD:
                    processed_lines[-1] = (
                        processed_lines[-1] + " " + current_combined.strip()
                    ).strip()
                else:
                    processed_lines.append(current_combined)
                current_combined = ""

            if not processed_lines:
                continue

            # 1. Encode all poem lines at once (huge speedup vs doing it line-by-line)
            line_embeddings = self.embedder.encode(
                processed_lines, convert_to_tensor=True, show_progress_bar=False
            )

            batch_similarities = util.cos_sim(
                line_embeddings, self.stacked_line_emotions
            )

            # Filter out boring/neutral lines to save processing time
            candidate_lines = []
            for idx, (line_text, line_emb, raw_scores) in enumerate(
                zip(processed_lines, line_embeddings, batch_similarities)
            ):
                line_score = [
                    s * ATTENUATION_FACTOR if s < LINE_EMOTION_THRESHOLD else s
                    for s in raw_scores.tolist()
                ]

                if max(line_score) >= LINE_EMOTION_THRESHOLD:
                    candidate_lines.append(
                        {
                            "idx": idx,
                            "text": line_text,
                            "score": line_score,
                            "emb": line_emb,
                        }
                    )

            # 2. Extract keywords using spaCy's optimized batch pipe
            if candidate_lines:
                candidate_texts = [item["text"] for item in candidate_lines]
                docs = self.nlp.pipe(candidate_texts, batch_size=64)

                for item, doc in zip(candidate_lines, docs):
                    classified_keywords = self.extract_keywords(doc, item["score"])

                    if classified_keywords:
                        raw_str = f"{author}_{title}_{poem_hash}_{item['idx']}".lower()
                        clean_str = re.sub(r"[^\w\s]", "", raw_str).replace(" ", "_")
                        line_id = re.sub(r"_+", "_", clean_str)

                        # Queue up the validated line and its keywords for the DB
                        current_batch.append(
                            {
                                "author": author,
                                "line_text": item["text"],
                                "line_score": item["score"],
                                "keywords": [k["word"] for k in classified_keywords],
                                "keyword_score": [
                                    k["score"] for k in classified_keywords
                                ],
                                "keyword_color": [
                                    k["color"] for k in classified_keywords
                                ],
                                "id": line_id,
                                "embedding": item["emb"].tolist(),
                            }
                        )

                        # Flush to Neo4j if the batch is full
                        if len(current_batch) >= WRITE_BATCH_SIZE:
                            counter += 1
                            total_lines_ingested += len(current_batch)
                            self.batch_ingest_lines(
                                current_batch, session, counter, total_lines_ingested
                            )
                            current_batch = []

        # Flush anything remaining after the loop finishes
        if current_batch:
            counter += 1
            total_lines_ingested += len(current_batch)
            self.batch_ingest_lines(
                current_batch, session, counter, total_lines_ingested
            )

    def add_frequency_to_keywords(self, session: neo4j.Session):
        # Calculates a TF-IDF style metric so we know which keywords are rare/valuable vs common
        query = """
        MATCH ()-[r:HAS_KEYWORD]->()
        WITH count(r) AS totalCount
        MATCH (n)-[:HAS_KEYWORD]->(k:PoemKeyword)
        WITH k, count(n) AS relativeFrequency, totalCount
        SET k.inverseFrequency = log(totalCount / toFloat(relativeFrequency))
        """
        session.run(query)

    def create_indexes(self, session: neo4j.Session):
        session.run(
            "CREATE CONSTRAINT line_id_unique IF NOT EXISTS FOR (l:Line) REQUIRE l.id IS UNIQUE"
        )
        session.run("CREATE INDEX IF NOT EXISTS FOR (k:PoemKeyword) ON (k.text)")
        session.run("CREATE INDEX IF NOT EXISTS FOR (a:Author) ON (a.name)")


def main():
    load_dotenv()

    pipeline = PoetryGraphPipeline()

    print("Connecting to driver")
    with get_neo4j_driver() as driver:
        print("Starting Neo4j session...")
        with driver.session() as session:
            print("Creating indexes and constraints...")
            pipeline.create_indexes(session)
            print("Inserting data into Neo4j...")
            pipeline.ingest_data(session)
            print("Adding frequency ratios to keywords...")
            pipeline.add_frequency_to_keywords(session)

    print("Ingestion complete!")


if __name__ == "__main__":
    main()
