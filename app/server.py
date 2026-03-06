import os

from flask import Flask, jsonify, render_template, request

from app.demo import generate_demo_response


app = Flask(__name__, template_folder="../templates")


def is_truthy(value: str) -> bool:
    return value.strip().lower() in {"1", "true", "yes", "on"}


DEMO_MODE = is_truthy(os.getenv("DEMO_MODE", "false"))
rag = None

if not DEMO_MODE:
    from app.config import get_settings
    from app.rag import PlaceRAG

    settings = get_settings()
    rag = PlaceRAG(settings)


@app.get("/")
def index():
    return render_template("index.html", demo_mode=DEMO_MODE)


@app.post("/recommend")
def recommend():
    payload = request.get_json(silent=True) or {}
    query = (payload.get("query") or "").strip()
    location = (payload.get("location") or "").strip()
    day = (payload.get("day") or "").strip()
    time_str = (payload.get("time") or "").strip()

    if not query:
        return jsonify({"success": False, "error": "Please include a query."}), 400

    try:
        if DEMO_MODE:
            result = generate_demo_response(
                query=query,
                location=location,
                day=day,
                time_str=time_str,
            )
        else:
            result = rag.recommend(query=query, location=location, day=day, time_str=time_str)

        return jsonify({"success": True, "recommendations": result, "demo_mode": DEMO_MODE})
    except Exception as exc:
        return jsonify({"success": False, "error": str(exc)}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002, debug=True)
