import os
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv
from neo4j import GraphDatabase

load_dotenv()

app = Flask(__name__)

# Read connection info from environment variables
NEO4J_URI = os.getenv("NEO4J_URI")
NEO4J_USER = os.getenv("NEO4J_USER")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD")

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

@app.route("/results", methods=["GET", "POST"])
def results():
    return render_template("results.html")
