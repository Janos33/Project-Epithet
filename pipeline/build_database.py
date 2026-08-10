"""
Poetry Graph Ingestion Pipeline
-------------------------------
This script reads a JSON corpus of poetry, extracts emotional keywords using
NLP (spaCy) and SentenceTransformers, and ingests the data into a Neo4j graph database.
"""

import datetime
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


def process_embeddings() -> dict:
    EMOTIONAL_EMBEDDINGS = {}

    for emotion, words in EMOTIONAL_ANCHORS.items():
        # Encode all words for an emotion, then average them into a single vector (centroid)
        word_embeddings = embedder.encode(words, convert_to_tensor=True)
        EMOTIONAL_EMBEDDINGS[emotion] = torch.mean(word_embeddings, dim=0)

    return EMOTIONAL_EMBEDDINGS


embedder = SentenceTransformer(EMBEDDER_MODEL)
nlp = spacy.load(SPACY_MODEL, disable=["parser", "ner"])
EMOTIONAL_EMBEDDINGS = process_embeddings()


def batch_ingest_lines(
    line_batch: list[dict], session, counter: int, total_lines_ingested: int
):
    """
    Executes a single Cypher transaction to ingest a batch of lines,
    authors, and keywords into Neo4j.
    """

    query = """
    // Unwind the python list of dictionaries into individual rows
    UNWIND $batch AS row

    // 1. Create or match the Author
    MERGE (w:Author {name: row.author})

    // 2. Create the Line (using row.id to ensure uniqueness)
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

    // Link Author to Line
    MERGE (w)-[:AUTHORED]->(l)

    // 3. Unwind parallel keyword arrays by index to process each word
    WITH l, row
    WHERE row.keywords IS NOT NULL AND size(row.keywords) > 0
    UNWIND range(0, size(row.keywords) - 1) AS i

    WITH l, 
        row.keywords[i] AS kw_word, 
        row.keyword_score[i] AS kw_score, 
        row.keyword_color[i] AS kw_color

    // 4. Merge the keyword node and attach base emotional properties
    MERGE (k:PoemKeyword {text: kw_word})
    ON CREATE SET 
        k.score = kw_score,
        k.color = kw_color
    ON MATCH SET
        k.score = kw_score,
        k.color = kw_color

    // 5. Connect the Line to its Keyword
    MERGE (l)-[:HAS_KEYWORD]->(k)  
    """

    session.run(query, batch=line_batch)
    print(
        f"[{datetime.datetime.now()}] Ingested batch #{counter} into Neo4j - batch size: {len(line_batch)} - ingested lines: {total_lines_ingested}"
    )


@lru_cache(maxsize=LRU_CACHE_SIZE)
def classify_word_emotional_tone(text: str) -> list:
    keyword_vec = embedder.encode(text, convert_to_tensor=True)

    scores = []

    # Check similarity against all base emotions
    for emotion, category_vec in EMOTIONAL_EMBEDDINGS.items():
        similarity = util.cos_sim(keyword_vec, category_vec).item()
        scores.append(similarity)

    # Return structured dict based on threshold

    scores = [
        score * ATTENUATION_FACTOR if score < WORD_EMOTION_THRESHOLD else score
        for score in scores
    ]

    return scores


def classify_line_emotional_tone(text: str) -> list:
    keyword_vec = embedder.encode(text, convert_to_tensor=True)

    scores = []

    # Check similarity against all base emotions
    for emotion, category_vec in EMOTIONAL_EMBEDDINGS.items():
        similarity = util.cos_sim(keyword_vec, category_vec).item()
        scores.append(similarity)

    # Return structured dict based on threshold

    scores = [
        score * ATTENUATION_FACTOR if score < LINE_EMOTION_THRESHOLD else score
        for score in scores
    ]

    return scores


def extract_keywords(text: str, line_score: list) -> dict:
    doc = nlp(text)
    keywords = []

    for token in doc:
        # Keep non-stopword Nouns, Verbs and Adjectives that are purely alphabetical
        if (
            token.pos_ in {"NOUN", "ADJ", "VERB"}
            and not token.is_stop
            and token.is_alpha
        ):
            keywords.append(token.lemma_.lower())

    keyword_classifications = []
    for word_str in keywords:
        word_score = classify_word_emotional_tone(word_str)

        scores_similarities = np.array(word_score) * np.array(line_score)

        if (
            max(word_score) > WORD_EMOTION_THRESHOLD
            and scores_similarities.max() > SCORE_SIMILARITY_THRESHOLD
        ):
            # Reduce the score for words that are below the threshold to avoid overemphasizing weak emotional signals

            color = get_blended_color(word_score)
            keyword_classifications.append(
                {"word": word_str, "score": word_score, "color": color}
            )

    return keyword_classifications


def get_blended_color(scores: list) -> str:
    # Extract emotion names, skipping 'Neutral' at index 0
    emotions = list(COLOR_MAP.keys())[1:]
    scores = np.array(scores, dtype=float)

    if scores.size > 2:
        top_indices = np.argsort(scores)[-2:]
        mask = np.zeros(scores.shape, dtype=bool)
        mask[top_indices] = True
        scores[~mask] = 0.0

    total_score = scores.sum()
    if total_score == 0:
        return COLOR_MAP.get("Neutral", "#A0A0A0")

    # Normalize weights so they sum to 1.0
    weights = scores / total_score

    r_sq_sum, g_sq_sum, b_sq_sum = 0.0, 0.0, 0.0

    for emotion, weight in zip(emotions, weights):
        if weight > 0 and emotion in COLOR_MAP:
            hex_str = COLOR_MAP[emotion].lstrip("#")
            r = int(hex_str[0:2], 16)
            g = int(hex_str[2:4], 16)
            b = int(hex_str[4:6], 16)

            # Accumulate squared channels for linear space blending
            r_sq_sum += weight * (r**2)
            g_sq_sum += weight * (g**2)
            b_sq_sum += weight * (b**2)

    # Take square root to return to sRGB space
    final_r = np.sqrt(r_sq_sum)
    final_g = np.sqrt(g_sq_sum)
    final_b = np.sqrt(b_sq_sum)

    return "#{:02x}{:02x}{:02x}".format(
        int(round(final_r)), int(round(final_g)), int(round(final_b))
    ).upper()


def ingest_data(session: neo4j.Session):
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

        # Treat the title as the first line of the poem for analysis
        lines_to_process = title + "\n" + text

        current_combined = ""
        processed_lines = []

        # 1. Line Segmentation Logic
        # Poetry lines can be awkward (too short or way too long).
        # This block normalizes them for better database readability.
        for raw_line in lines_to_process.splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            # Split excessively long lines by punctuation
            if len(raw_line) > MAX_LINE_LENGTH:
                sub_lines = [
                    s.strip() for s in re.split(r"(?<=[,;:]);?", raw_line) if s.strip()
                ]
            else:
                sub_lines = [raw_line]

            # Combine very short fragments together
            for line in sub_lines:
                if len(line) < LINE_LENGTH_THRESHOLD:
                    current_combined += (" " + line) if current_combined else line
                else:
                    if current_combined:
                        if len(current_combined) >= LINE_LENGTH_THRESHOLD:
                            processed_lines.append(current_combined)
                        current_combined = ""
                    processed_lines.append(line)

        # Catch any remaining combined text at the end of the poem
        if current_combined:
            if len(current_combined) >= LINE_LENGTH_THRESHOLD:
                processed_lines.append(current_combined)
            current_combined = ""

        # 2. Extract Data & Queue Batch
        for idx, line_text in enumerate(processed_lines):
            line_score = classify_line_emotional_tone(line_text)

            if max(line_score) >= LINE_EMOTION_THRESHOLD:
                classified_keywords = extract_keywords(line_text, line_score)

                if classified_keywords:
                    # Generate a unique deterministic ID for the line

                    raw_str = f"{author}_{title}_{idx}".lower()
                    clean_str = re.sub(r"[^\w\s]", "", raw_str).replace(" ", "_")
                    line_id = re.sub(r"_+", "_", clean_str)

                    # Note: 'embedding' is left as None here and calculated in bulk later
                    current_batch.append(
                        {
                            "author": author,
                            "line_text": line_text,
                            "line_score": line_score,
                            "keywords": [k["word"] for k in classified_keywords],
                            "keyword_score": [k["score"] for k in classified_keywords],
                            "keyword_color": [k["color"] for k in classified_keywords],
                            "id": line_id,
                            "embedding": None,
                        }
                    )

                    # 3. Batch Vectorization & Ingestion
                    # When the batch fills up, run embeddings in parallel rather than a loop
                    if len(current_batch) >= WRITE_BATCH_SIZE:
                        line_texts = [item["line_text"] for item in current_batch]

                        # Batch encoding utilizes CPU/GPU matrix multiplication for speed
                        embeddings = embedder.encode(
                            line_texts, batch_size=128, show_progress_bar=False
                        )

                        # Map embeddings back to their dict records
                        for item, emb in zip(current_batch, embeddings):
                            item["embedding"] = emb.tolist()

                        counter += 1
                        total_lines_ingested += len(current_batch)
                        batch_ingest_lines(
                            current_batch, session, counter, total_lines_ingested
                        )
                        current_batch = []

    # Final cleanup flush for any lines remaining after the last poem
    if current_batch:
        line_texts = [item["line_text"] for item in current_batch]
        embeddings = embedder.encode(
            line_texts, batch_size=128, show_progress_bar=False
        )

        for item, emb in zip(current_batch, embeddings):
            item["embedding"] = emb.tolist()

        counter += 1
        total_lines_ingested += len(current_batch)
        batch_ingest_lines(current_batch, session, counter, total_lines_ingested)
        current_batch = []


def add_frequency_to_keywords(session: neo4j.Session):
    query = """
    MATCH (n)-[]->(k:PoemKeyword)
    WITH k, count(n) AS relativeFrequency
    WITH collect({node: k, count: relativeFrequency}) AS keywords, sum(relativeFrequency) AS totalCount
    UNWIND keywords AS item
    SET item.node.inverseFrequency = log(totalCount / toFloat(item.count))
    """

    session.run(query)


def create_indexes(session: neo4j.Session):
    session.run(
        "CREATE CONSTRAINT line_id_unique IF NOT EXISTS FOR (l:Line) REQUIRE l.id IS UNIQUE"
    )

    session.run("CREATE INDEX IF NOT EXISTS FOR (k:PoemKeyword) ON (k.text)")
    session.run("CREATE INDEX IF NOT EXISTS FOR (a:Author) ON (a.name)")


def main():
    load_dotenv()

    print("Connecting to driver")
    with get_neo4j_driver() as driver:
        print("Starting Neo4j session...")
        with driver.session() as session:
            print("Creating indexes and constraints...")
            create_indexes(session)
            print("Inserting data into Neo4j...")
            ingest_data(session)
            print("Adding frequency ratios to keywords...")
            add_frequency_to_keywords(session)

    print("Ingestion complete!")


if __name__ == "__main__":
    main()
