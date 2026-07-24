import json
import os
import re

from neo4j import GraphDatabase
from sentence_transformers import SentenceTransformer
from init import driver
import spacy

# --- Settings ---
LINE_LENGTH_THRESHOLD = 15
MAX_LINE_LENGTH = 50

# Model Initializations
# 1. Vector Embedder (~384 dimensions; fast and dense)
embedder = SentenceTransformer("all-MiniLM-L6-v2")

# 2. NLP Pipeline for Keyword Extraction
# Make sure to run `python -m spacy download en_core_web_sm` first!
nlp = spacy.load("en_core_web_sm")

# --- Helper Functions ---


def extract_keywords(text: str) -> list[str]:
    """Extracts nouns and adjectives as 'oomph' keywords, normalized to lowercase lemmas."""
    doc = nlp(text)
    keywords = []
    for token in doc:
        # Keep non-stopword Nouns and Adjectives
        if token.pos_ in {"NOUN", "ADJ"} and not token.is_stop and token.is_alpha:
            keywords.append(token.lemma_.lower())
    return list(set(keywords))  # Deduplicate per line


# --- Main Pipeline ---

# Initialize Neo4j Driver
with open("data/poems.json", "r", encoding="utf-8") as f:
    des_data = json.load(f)

with driver.session() as session:
    for poem in des_data:
        author = poem.get("Author", "Unknown")
        title = poem.get("Title", "Untitled")
        text = poem.get("text", "")

        lines_to_process = title + "\n" + text

        current_combined = ""
        processed_lines = []

        # 1. Line Segmentation Logic
        for raw_line in lines_to_process.splitlines():
            raw_line = raw_line.strip()
            if not raw_line:
                continue

            if len(raw_line) > MAX_LINE_LENGTH:
                sub_lines = [
                    s.strip()
                    for s in re.split(r"(?<=[,;:]);?", raw_line)
                    if s.strip()
                ]
            else:
                sub_lines = [raw_line]

            for line in sub_lines:
                if len(line) < LINE_LENGTH_THRESHOLD:
                    current_combined += (" " + line) if current_combined else line
                else:
                    if current_combined:
                        if len(current_combined) >= LINE_LENGTH_THRESHOLD:
                            processed_lines.append(current_combined)
                        current_combined = ""
                    processed_lines.append(line)

        if current_combined:
            if len(current_combined) >= LINE_LENGTH_THRESHOLD:
                processed_lines.append(current_combined)
            current_combined = ""

        # 2. Neo4j Batch Node & Relationship Ingestion
        for idx, line_text in enumerate(processed_lines):
            # A. Generate Vector Embedding
            embedding = embedder.encode(line_text).tolist()

            # B. Extract Keywords via spaCy
            keywords = extract_keywords(line_text)

            # C. Unique Line ID strategy (Poem Title + Line Index)
            line_id = (
                f"{author}_{title}_{idx}".lower().replace(" ", "_")[
                    :100
                ]  # Sanitize string ID
            )

            # D. Execute Cypher Transaction
            query = """
            // 1. Merge Writer
            MERGE (w:Writer {name: $author})

            // 2. Merge Line with text and embedding vector
            MERGE (l:Line {id: $line_id})
            ON CREATE SET 
                l.text = $line_text,
                l.embedding = $embedding

            // 3. Connect Writer to Line
            MERGE (w)-[:AUTHORED]->(l)

            // 4. Merge Keywords and connect to Line
            WITH l
            UNWIND $keywords AS kw_text
            MERGE (k:PoemKeyword {word: kw_text})
            MERGE (l)-[:HAS_KEYWORD]->(k)
            """

            session.run(
                query,
                author=author,
                line_id=line_id,
                line_text=line_text,
                embedding=embedding,
                keywords=keywords,
            )

            print(
                f"Ingested line [{idx + 1}/{len(processed_lines)}] for '{title}' with {len(keywords)} keywords."
            )

driver.close()
print("Ingestion complete!")