import os
from neo4j import GraphDatabase

# --- Helper Functions (used by Flask app.py) ---


def add_person(driver, name):
    """Creates a Person node in Neo4j."""
    query = "MERGE (p:Person {name: $name}) RETURN p.name AS name"
    records, _, _ = driver.execute_query(query, name=name)
    return records[0]["name"] if records else None


def get_all_people(driver):
    """Retrieves all Person nodes from Neo4j."""
    query = "MATCH (p:Person) RETURN p.name AS name"
    records, _, _ = driver.execute_query(query)
    return [record["name"] for record in records]


def create_knows_relationship(driver, name1, name2):
    """Connects two Person nodes with a KNOWS relationship."""
    query = """
    MATCH (a:Person {name: $name1})
    MATCH (b:Person {name: $name2})
    MERGE (a)-[r:KNOWS]->(b)
    RETURN a.name AS sender, b.name AS receiver
    """
    records, _, _ = driver.execute_query(query, name1=name1, name2=name2)
    return records


# --- Standalone Test Execution ---
if __name__ == "__main__":
    # Load credentials for direct test running
    URI = os.getenv("NEO4J_URI") or "bolt://localhost:7687"
    USER = os.getenv("NEO4J_USER") or "neo4j"
    PASSWORD = os.getenv("NEO4J_PASSWORD") or "secure_pass"

    print("Connecting to Neo4j for testing...")
    with GraphDatabase.driver(URI, auth=(USER, PASSWORD)) as test_driver:
        # Run test queries
        print("Adding Alice...", add_person(test_driver, "Alice"))
        print("Adding Bob...", add_person(test_driver, "Bob"))
        print(
            "Creating relationship...",
            create_knows_relationship(test_driver, "Alice", "Bob"),
        )
        print("Current people in DB:", get_all_people(test_driver))