from __future__ import annotations

import json
from pathlib import Path
from typing import Any


def load_demo_catalog() -> dict[str, Any]:
    path = Path(__file__).resolve().parents[1] / "data" / "demo_responses.json"
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def _matches(query: str, keywords: list[str]) -> bool:
    q = query.lower()
    return any(k.lower() in q for k in keywords)


def generate_demo_response(query: str, location: str = "", day: str = "", time_str: str = "") -> str:
    catalog = load_demo_catalog()

    chosen = catalog.get("default", {})
    for bucket in catalog.get("buckets", []):
        if _matches(query, bucket.get("keywords", [])):
            chosen = bucket
            break

    title = chosen.get("title", "Top places")
    picks = chosen.get("picks", [])

    lines = [f"{title} (Demo Mode)\n"]

    for pick in picks[:3]:
        lines.append(
            f"{pick['name']} - {pick['area']} ({pick['hours']}):\n"
            f"     insight: {pick['why']}\n"
        )

    lines.append("Note: Demo mode is enabled. Results are sample recommendations, not live model output.")
    
    return "\n".join(lines)
