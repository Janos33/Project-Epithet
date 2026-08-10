from engine import load_dataset, find_keywords
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv

load_dotenv()

app = Flask(__name__)

# Load data at start
print("Initializing engine and loading 800k dataset into memory...")
RAW_EMBEDDINGS, CLUSTER_LABELS, METADATA = load_dataset()
print("Engine ready!")

@app.route("/", methods=["GET"])
def index():
    # Renders your index.html form
    return render_template("index.html")

@app.route("/results", methods=["POST"])
def results():
    # 1. Check textarea first
    poem_text = request.form.get("poem_text")
    author_name = request.form.get("author_name")
    
    # 2. Guard against completely empty submissions
    if not poem_text or not poem_text.strip():
        return render_template("results.html", keywords=[])

    # 3. Run pipeline
    results = find_keywords(poem_text, RAW_EMBEDDINGS, CLUSTER_LABELS, METADATA)

    if results is None:
        results = []
    
    return render_template("results.html", keywords=results, author = author_name)

if __name__ == "__main__":
    app.run(debug=True, port=5000)