import json
from engine import PoemKeywordExtractor
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv
import parameters

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB
app.config["MAX_FORM_MEMORY_SIZE"] = 8 * 1024 * 1024  # 8 MB

print("Initializing engine and loading dataset into memory...")
extractor = PoemKeywordExtractor()
print("Engine ready!")


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/results", methods=["POST"])
def results():
    # 1. Grab the mathematically encoded embeddings from the frontend
    client_embeddings_raw = request.form.get("client_poem_embeddings")

    weights = parameters.Weights(
        emotional_weight=int(request.form.get("emotional_weight")) / 100,
        line_weight=int(request.form.get("line_weight")) / 100,
        context_line_amount=int(request.form.get("context_line_amount")),
        neighborhood_amount=int(request.form.get("neighborhood_amount")),
        max_results=int(request.form.get("max_results")),
    )

    # 2. Guard against completely empty submissions or bypassing the JS
    if not client_embeddings_raw:
        return render_template("index.html", keywords=[])

    # Convert the JSON string representation back into a Python list
    poem_embeddings = json.loads(client_embeddings_raw)

    # 3. Run pipeline using the pre-calculated embeddings
    results = extractor.extract(poem_embeddings, weights)

    if results is None:
        results = []

    return render_template("index.html", keywords=results)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
