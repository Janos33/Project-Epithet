import os
import time
import uuid
import logging
import threading
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer

from engine import PoemKeywordExtractor
from parameters import Weights, list_profiles, load_profile

load_dotenv(override=True)

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB
app.config["MAX_FORM_MEMORY_SIZE"] = 8 * 1024 * 1024  # 8 MB

# --- Load one extractor per active profile ---
_configured_profiles = [
    name.strip() for name in os.getenv("PROFILES", "Default").split(",") if name.strip()
]
_available_profiles = set(list_profiles())

for name in _configured_profiles:
    if name not in _available_profiles:
        logger.warning(
            "Profile '%s' is configured but not found under data/profiles/ -- skipping.",
            name,
        )

print("Initializing engine and loading dataset(s) into memory...")

extractors: dict[str, PoemKeywordExtractor] = {}
profile_info: dict[str, dict] = {}
_embedders_by_model: dict[str, SentenceTransformer] = {}

for name in _configured_profiles:
    if name not in _available_profiles:
        continue
    try:
        print(f"Attempting to load profile '{name}'.")
        profile = load_profile(name)

        embedder = _embedders_by_model.get(profile.embedder_model)
        if embedder is None:
            embedder = SentenceTransformer(profile.embedder_model)
            _embedders_by_model[profile.embedder_model] = embedder

        extractors[name] = PoemKeywordExtractor(profile, embedder=embedder)
        profile_info[name] = {
            "id": name,
            "name": profile.display_name,
            "description": profile.description,
        }
        print(f"Loaded profile '{name}'.")
    except Exception:
        logger.exception("Profile '%s' failed to load -- skipping.", name)

if not extractors:
    raise RuntimeError(
        "No profiles loaded successfully. "
        f"Configured (PROFILES): {', '.join(_configured_profiles) or '(none)'}. "
        f"Found under data/profiles/: {', '.join(sorted(_available_profiles)) or '(none)'}."
    )

DEFAULT_PROFILE = next(iter(extractors)) if len(extractors) == 1 else None

print(f"Engine ready! Active profiles: {', '.join(extractors)}")

queue_lock = threading.Lock()
active_queue = {}


def clean_stale_tickets():
    now = time.time()
    with queue_lock:
        stale_ids = []
        for tid, data in active_queue.items():
            time_since_seen = now - data["last_seen"]
            is_stale_waiting = data.get("status") == "waiting" and time_since_seen > 10
            is_stale_processing = (
                data.get("status") == "processing" and time_since_seen > 300
            )

            if is_stale_waiting or is_stale_processing:
                stale_ids.append(tid)

        for tid in stale_ids:
            del active_queue[tid]


def _remove_ticket(ticket_id):
    with queue_lock:
        if ticket_id in active_queue:
            del active_queue[ticket_id]


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/results", methods=["GET"])
def results():
    return render_template("index.html", keywords=[])


@app.route("/api/profiles", methods=["GET"])
def profiles():
    return jsonify(
        {
            "profiles": list(profile_info.values()),
            "default": DEFAULT_PROFILE,
        }
    )


@app.route("/api/queue/join", methods=["POST"])
def queue_join():
    clean_stale_tickets()
    ticket_id = str(uuid.uuid4())
    now = time.time()

    with queue_lock:
        active_queue[ticket_id] = {
            "joined_at": now,
            "last_seen": now,
            "status": "waiting",
        }
        position = list(active_queue.keys()).index(ticket_id)

    return jsonify({"ticket_id": ticket_id, "position": position})


@app.route("/api/queue/status/<ticket_id>", methods=["GET"])
def queue_status(ticket_id):
    clean_stale_tickets()
    now = time.time()

    with queue_lock:
        if ticket_id not in active_queue:
            return jsonify({"status": "expired", "position": -1}), 404

        active_queue[ticket_id]["last_seen"] = now
        position = list(active_queue.keys()).index(ticket_id)

    status = "ready" if position == 0 else "waiting"
    return jsonify({"status": status, "position": position})


@app.route("/api/queue/process", methods=["POST"])
def queue_process():
    clean_stale_tickets()
    data = request.get_json() or {}
    ticket_id = data.get("ticket_id")

    with queue_lock:
        if not ticket_id or ticket_id not in active_queue:
            return jsonify({"error": "Invalid or expired ticket"}), 403

        position = list(active_queue.keys()).index(ticket_id)
        if position != 0:
            return jsonify({"error": "Not your turn in queue"}), 429

        active_queue[ticket_id]["status"] = "processing"
        active_queue[ticket_id]["last_seen"] = time.time()

    profile_name = data.get("profile") or DEFAULT_PROFILE
    extractor = extractors.get(profile_name)
    if extractor is None:
        _remove_ticket(ticket_id)
        return jsonify(
            {
                "error": "Unknown or missing profile",
                "available_profiles": list(extractors),
            }
        ), 400

    poem_embeddings = data.get("embeddings")

    weights = Weights(
        emotional_weight=float(data.get("emotional_weight", 60)) / 100,
        line_weight=float(data.get("line_weight", 85)) / 100,
        context_line_amount=int(data.get("context_line_amount", 1000)),
        neighborhood_amount=int(data.get("neighborhood_amount", 500)),
        max_results=int(data.get("max_results", 20)),
    )

    results = extractor.extract(poem_embeddings, weights)
    if results is None:
        results = []

    _remove_ticket(ticket_id)
    return jsonify(results)


@app.route("/favicon.ico", methods=["GET"])
def favicon():
    return app.send_static_file("favicons/favicon.ico")


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000, extra_files=[".env"])
