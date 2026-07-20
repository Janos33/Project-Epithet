import os
from flask import Flask, jsonify, render_template
from neo4j import GraphDatabase

import db

app = Flask(__name__)

# Read connection info from environment variables
NEO4J_URI = os.getenv("NEO4J_URI") or "bolt://neo4j:7687"
NEO4J_USER = os.getenv("NEO4J_USER") or "neo4j"
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD") or "your_password"

# Initialize the Neo4j driver
driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))


# Helper function to close driver on app shutdown
@app.teardown_appcontext
def close_driver(exception):
    pass  # Keeps connection open during normal Flask worker lifecycle

# Render the html test page at the root URL
@app.route("/")
def index():
    return render_template("index.html")