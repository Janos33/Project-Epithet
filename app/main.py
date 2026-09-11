import json
import time
import uuid
import threading
from flask import Flask, jsonify, render_template, request
from dotenv import load_dotenv

from engine import PoemKeywordExtractor
import parameters

load_dotenv()

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB
app.config["MAX_FORM_MEMORY_SIZE"] = 8 * 1024 * 1024  # 8 MB

print("Initializing engine and loading dataset into memory...")
extractor = PoemKeywordExtractor()
print("Engine ready!")

# Use a lock to prevent Thread race conditions in Flask when modifying the queue
queue_lock = threading.Lock()

# Active queue memory: { ticket_id: { "joined_at": float, "last_seen": float, "status": str } }
active_queue = {}


def clean_stale_tickets():
    """
    Purges tickets inactive for >10 seconds, UNLESS they are actively processing.
    Also clears stuck processing tickets if they've been running for > 5 minutes (timeout failsafe).
    """
    now = time.time()

    with queue_lock:
        stale_ids = []
        for tid, data in active_queue.items():
            time_since_seen = now - data["last_seen"]

            # If waiting and stopped polling for 10s
            is_stale_waiting = data.get("status") == "waiting" and time_since_seen > 10
            # If processing but hung for more than 300s (5 mins)
            is_stale_processing = (
                data.get("status") == "processing" and time_since_seen > 300
            )

            if is_stale_waiting or is_stale_processing:
                stale_ids.append(tid)

        for tid in stale_ids:
            del active_queue[tid]


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html")


@app.route("/results", methods=["GET"])
def results():
    return render_template("index.html", keywords=[])


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
        # Position is simply their index in the dictionary
        position = list(active_queue.keys()).index(ticket_id)

    return jsonify({"ticket_id": ticket_id, "position": position})


@app.route("/api/queue/status/<ticket_id>", methods=["GET"])
def queue_status(ticket_id):
    clean_stale_tickets()
    now = time.time()

    with queue_lock:
        if ticket_id not in active_queue:
            return jsonify({"status": "expired", "position": -1}), 404

        # Update last seen since they are actively polling
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

    poem_embeddings = data.get("embeddings")

    weights = parameters.Weights(
        emotional_weight=float(data.get("emotional_weight", 50)) / 100,
        line_weight=float(data.get("line_weight", 50)) / 100,
        context_line_amount=int(data.get("context_line_amount", 3)),
        neighborhood_amount=int(data.get("neighborhood_amount", 5)),
        max_results=int(data.get("max_results", 20)),
    )

    results = extractor.extract(poem_embeddings, weights)
    if results is None:
        results = []
    # -----------------------------------------------------

    with queue_lock:
        if ticket_id in active_queue:
            del active_queue[ticket_id]

    return jsonify(results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", debug=True, port=5000)
