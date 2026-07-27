"""
Poetry Graph Ingestion Pipeline
-------------------------------
This script reads a JSON corpus of poetry, extracts emotional keywords using 
NLP (spaCy) and SentenceTransformers, and ingests the data into a Neo4j graph database.
"""

import json
import os
import re
from dotenv import load_dotenv
from neo4j import GraphDatabase, Driver
from sentence_transformers import SentenceTransformer, util
from functools import lru_cache
import spacy
import torch

# --- Settings & Configuration ---

# Thresholds for breaking up or combining lines of poetry
LINE_LENGTH_THRESHOLD = 15
MAX_LINE_LENGTH = 50

# How many lines to accumulate in memory before sending to Neo4j
BATCH_SIZE = 2000

embedder = SentenceTransformer("all-MiniLM-L6-v2")
nlp = spacy.load("en_core_web_sm", disable=["parser", "ner"])

EMOTIONAL_ANCHORS = {
    "Radiance": [
        "glow", "radiance", "joy", "laughter", "hope", "dawn", "promise", "blossom", "sunshine", "holy", "cupcake", "bunny"
    ],
    "Serenity": [
        "stillness", "silence", "calm", "infinity", "cosmos", "eternal", "wonder", "breeze", "timeless", "awe"
    ],
    "Passion": [
        "passion", "desire", "burning", "obsession", "tender", "heartbeat", "yearning", "memory", "touch"
    ],
    "Melancholy": [
        "grief", "loneliness", "mourning", "regret", "guilt", "shame", "sorrow", "yesterday", "absence", "memory", "bittersweet"
    ],
    "Torment": [
        "dread", "fear", "rage", "storm", "shatter", "pain", "broken", "desolation", "ruin", "unfair", "unjust", "poverty"
    ],
    "Delirium": [
        "madness", "insanity", "frenzy", "chaos", "eerie", "haunting", "shadow", "disgust", "abyss", "mystery"
    ],
    "Transience": [
        "time", "fading", "dust", "fleeting", "autumn", "mortality", "ephemeral", "passing", "vanishing", "wither", "earth", "physical", "plain", "ordinary", "fact"
    ]
}

COLOR_MAP = {
    "Neutral": "#A0A0A0",       # Gray
    "Radiance": "#F4D35E",      # Bright Amber-Yellow (Joy, Hope, Light)
    "Serenity": "#3B7A57",      # Sage Green (Peace, Transcendence)
    "Passion": "#9B1D20",       # Crimson Red (Love, Yearning, Obsession)
    "Melancholy": "#3D5A80",    # Steel Blue (Sorrow, Nostalgia, Guilt, Shame)
    "Torment": "#D95D39",       # Burnt Orange (Tension, Despair, Fear)
    "Delirium": "#5E3A6D",      # Deep Purple/Magenta (Madness, Dread, Disgust, Mystery)
    "Transience": "#8B4513"      # Dark Brown (Time, Fading, Ephemeral)
}

EMOTIONAL_EMBEDDINGS = {}

def get_neo4j_driver() -> Driver:
    NEO4J_URI = os.getenv("NEO4J_URI")
    NEO4J_USER = os.getenv("NEO4J_USER")
    NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")
    return GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))

def process_embeddings():
    for emotion, words in EMOTIONAL_ANCHORS.items():
            # Encode all words for an emotion, then average them into a single vector (centroid)
            word_embeddings = embedder.encode(words, convert_to_tensor=True)
            EMOTIONAL_EMBEDDINGS[emotion] = torch.mean(word_embeddings, dim=0)

def batch_ingest_lines(line_batch: list[dict], session):
    """
    Executes a single Cypher transaction to ingest a batch of lines, 
    authors, and keywords into Neo4j.
    """
    print(f"Ingesting batch of {len(line_batch)} lines into Neo4j...")
    
    query = """
    // Unwind the python list of dictionaries into individual rows
    UNWIND $batch AS row

    // 1. Create or match the Author
    MERGE (w:Author {name: row.author})

    // 2. Create the Line (using row.id to ensure uniqueness)
    MERGE (l:Line {id: row.id})
    ON CREATE SET 
        l.text = row.line_text,
        l.embedding = row.embedding
    ON MATCH SET
        l.text = row.line_text,
        l.embedding = row.embedding

    // Link Author to Line
    MERGE (w)-[:AUTHORED]->(l)

    // 3. Unwind parallel keyword arrays by index to process each word
    WITH l, row
    WHERE row.keywords IS NOT NULL AND size(row.keywords) > 0
    UNWIND range(0, size(row.keywords) - 1) AS i

    WITH l, 
        row.keywords[i] AS kw_word, 
        row.keyword_emotional_tone[i] AS kw_emotion, 
        row.keyword_score[i] AS kw_score, 
        row.keyword_color[i] AS kw_color

    // 4. Merge the keyword node and attach base emotional properties
    MERGE (k:PoemKeyword {word: kw_word})
    ON CREATE SET 
        k.emotion = kw_emotion,
        k.score = kw_score,
        k.color = kw_color
    ON MATCH SET
        k.emotion = kw_emotion,
        k.score = kw_score,
        k.color = kw_color

    // 5. Connect the Line to its Keyword
    MERGE (l)-[:HAS_KEYWORD]->(k)  
    """
    
    session.run(query, batch=line_batch)

@lru_cache(maxsize=10000)
def classify_emotional_tone(keyword: str, threshold: float = 0.5) -> dict:
    """
    Compares a single keyword against the predefined emotional centroids using cosine similarity.
    Returns the emotion if it passes the threshold, otherwise returns Neutral.
    """
    keyword_vec = embedder.encode(keyword, convert_to_tensor=True)

    best_score = 0.0
    best_emotion = "Neutral"

    # Check similarity against all base emotions
    for emotion, category_vec in EMOTIONAL_EMBEDDINGS.items():
        similarity = util.cos_sim(keyword_vec, category_vec).item()
        if similarity > best_score:
            best_score = similarity
            best_emotion = emotion

    # Return structured dict based on threshold
    if best_score < threshold:
        return {"emotion": "Neutral", "score": best_score, "color": COLOR_MAP["Neutral"]}
    else:
        return {
            "word": keyword,
            "emotion": best_emotion,
            "score": round(best_score, 3),
            "color": COLOR_MAP[best_emotion]
        }

def extract_keywords(text: str) -> dict:
    """
    Uses spaCy to parse text and extract meaningful nouns, verbs, and adjectives.
    Filters out neutral words to keep the graph focused on emotional language.
    """
    doc = nlp(text)
    keywords = []
    
    for token in doc:
        # Keep non-stopword Nouns, Verbs and Adjectives that are purely alphabetical
        if token.pos_ in {"NOUN", "ADJ", "VERB"} and not token.is_stop and token.is_alpha:
            keywords.append(token.lemma_.lower())

    keyword_classifications = []
    for word_str in keywords:
        classification = classify_emotional_tone(word_str)
        # Drop neutral words from the final dataset to prevent noise
        if classification["emotion"] != "Neutral":
            keyword_classifications.append(classification)
    
    return keyword_classifications

def ingest_data(driver: Driver):

    current_batch = []

    print("Loading dataset...")
    with open("data/raw/poems.json", "r", encoding="utf-8") as f:
        des_data = json.load(f)

    print("Starting Neo4j session...")
    with driver.session() as session:

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
                        s.strip()
                        for s in re.split(r"(?<=[,;:]);?", raw_line)
                        if s.strip()
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
                
                classified_keywords = extract_keywords(line_text)

                if classified_keywords:
                    
                    # Generate a unique deterministic ID for the line
                    
                    raw_str = f"{author}_{title}_{idx}".lower()
                    clean_str = re.sub(r'[^\w\s]', '', raw_str).replace(" ", "_")
                    line_id = re.sub(r'_+', '_', clean_str)

                    # Note: 'embedding' is left as None here and calculated in bulk later
                    current_batch.append({
                        "author": author,
                        "line_text": line_text,
                        "keywords": [k["word"] for k in classified_keywords],
                        "keyword_emotional_tone": [k["emotion"] for k in classified_keywords],
                        "keyword_score": [k["score"] for k in classified_keywords],
                        "keyword_color": [k["color"] for k in classified_keywords],
                        "id": line_id,
                        "embedding": None 
                    })

                    # 3. Batch Vectorization & Ingestion
                    # When the batch fills up, run embeddings in parallel rather than a loop
                    if len(current_batch) >= BATCH_SIZE:
                        
                        line_texts = [item["line_text"] for item in current_batch]
                        
                        # Batch encoding utilizes CPU/GPU matrix multiplication for speed
                        embeddings = embedder.encode(line_texts, batch_size=128, show_progress_bar=False)

                        # Map embeddings back to their dict records
                        for item, emb in zip(current_batch, embeddings):
                            item["embedding"] = emb.tolist()

                        batch_ingest_lines(current_batch, session)
                        current_batch = []

        # Final cleanup flush for any lines remaining after the last poem
        if current_batch:
            
            line_texts = [item["line_text"] for item in current_batch]
            embeddings = embedder.encode(line_texts, batch_size=128, show_progress_bar=False)
            
            for item, emb in zip(current_batch, embeddings):
                item["embedding"] = emb.tolist()
                
            batch_ingest_lines(current_batch, session)
            current_batch = []

def main():
    load_dotenv()

    print("Pre-computing emotional anchors...")
    process_embeddings()

    print("Connecting to driver")
    with get_neo4j_driver() as driver:
        print("Connected to the database")
        ingest_data(driver)

    print("Ingestion complete!")

if __name__ == "__main__":
    main()