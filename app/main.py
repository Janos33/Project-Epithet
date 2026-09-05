from engine import PoemKeywordExtractor
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv
import parameters

load_dotenv()

app = Flask(__name__)

# Load data at start
print("Initializing engine and loading dataset into memory...")
extractor = PoemKeywordExtractor()
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

    weights = parameters.Weights(
        emotional_weight=int(request.form.get("emotional_weight")) / 100,
        line_weight=int(request.form.get("line_weight")) / 100,
        context_line_amount=int(request.form.get("context_line_amount")),
        neighborhood_amount=int(request.form.get("neighborhood_amount")),
        max_results=int(request.form.get("max_results")),
    )

    # 2. Guard against completely empty submissions
    if not poem_text or not poem_text.strip():
        return render_template("index.html", keywords=[])

    # 3. Run pipeline
    results = extractor.extract(poem_text, weights)

    if results is None:
        results = []

    return render_template("index.html", keywords=results, author=author_name)


if __name__ == "__main__":
    app.run(debug=True, port=5000)
