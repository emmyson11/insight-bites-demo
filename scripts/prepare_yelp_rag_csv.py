#!/usr/bin/env python3
"""Prepare a RAG-ready CSV from Yelp JSON datasets.

Expected JSONL inputs (one JSON object per line):
- business.json (required)
- review.json (optional)
- tip.json (optional)
- user.json (optional; requires review.json to join)
- checkin.json (optional)

Outputs:
1) Tabular merged CSV
2) RAG-ready CSV with embedding_text
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build Yelp cafe CSV for RAG")
    parser.add_argument("--business", required=True, help="Path to business JSONL file")
    parser.add_argument("--review", help="Path to review JSONL file (optional)")
    parser.add_argument("--tip", help="Path to tip JSONL file (optional)")
    parser.add_argument("--user", help="Path to user JSONL file (optional)")
    parser.add_argument("--checkin", help="Path to checkin JSONL file (optional)")
    parser.add_argument(
        "--out-tabular",
        default="data/yelp_places_tabular.csv",
        help="Output merged tabular CSV path",
    )
    parser.add_argument(
        "--out-rag",
        default="data/yelp_places_for_rag.csv",
        help="Output RAG-ready CSV path",
    )
    parser.add_argument(
        "--max-review-snippets",
        type=int,
        default=3,
        help="Max review snippets to keep per business",
    )
    parser.add_argument(
        "--max-tip-snippets",
        type=int,
        default=3,
        help="Max tip snippets to keep per business",
    )
    return parser.parse_args()


def iter_json_lines(path: Path) -> Iterable[dict]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def coerce_categories(categories: object) -> str:
    if isinstance(categories, list):
        return ", ".join(str(x).strip() for x in categories if str(x).strip())
    if isinstance(categories, str):
        return categories.strip()
    return ""


def is_food_drink_place(categories_text: str) -> bool:
    if not categories_text:
        return False
    hay = categories_text.lower()
    keys = (
        "restaurant",
        "restaurants",
        "food",
        "coffee",
        "cafe",
        "cafes",
        "tea",
        "bar",
        "bars",
        "pub",
        "brewery",
        "bakery",
        "dessert",
        "brunch",
        "breakfast",
        "diner",
        "juice",
        "sandwich",
        "pizza",
        "sushi",
        "taco",
    )
    return any(k in hay for k in keys)


def load_businesses(path: Path) -> pd.DataFrame:
    rows: List[dict] = []
    for obj in iter_json_lines(path):
        categories_text = coerce_categories(obj.get("categories"))
        if not is_food_drink_place(categories_text):
            continue

        postal_code = obj.get("postal_code")
        if postal_code is None:
            # Some docs/sources show "postal code".
            postal_code = obj.get("postal code", "")

        rows.append(
            {
                "business_id": obj.get("business_id", ""),
                "name": obj.get("name", ""),
                "address": obj.get("address", ""),
                "city": obj.get("city", ""),
                "state": obj.get("state", ""),
                "postal_code": postal_code or "",
                "latitude": obj.get("latitude", ""),
                "longitude": obj.get("longitude", ""),
                "stars": obj.get("stars", ""),
                "review_count": obj.get("review_count", 0),
                "is_open": obj.get("is_open", ""),
                "categories": categories_text,
                "attributes": json.dumps(obj.get("attributes", {}), ensure_ascii=True),
                "hours": json.dumps(obj.get("hours", {}), ensure_ascii=True),
            }
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise ValueError("No eat/drink businesses found. Check category filter or input file.")
    return df


def aggregate_reviews(
    path: Path,
    target_ids: set[str],
    max_snippets: int,
) -> tuple[pd.DataFrame, Dict[str, List[str]], set[str]]:
    counts: Dict[str, int] = defaultdict(int)
    star_sums: Dict[str, float] = defaultdict(float)
    useful_sum: Dict[str, int] = defaultdict(int)
    funny_sum: Dict[str, int] = defaultdict(int)
    cool_sum: Dict[str, int] = defaultdict(int)
    snippets: Dict[str, List[str]] = defaultdict(list)
    biz_user_ids: Dict[str, List[str]] = defaultdict(list)
    all_user_ids: set[str] = set()

    for obj in iter_json_lines(path):
        bid = obj.get("business_id")
        if bid not in target_ids:
            continue

        counts[bid] += 1
        star_sums[bid] += float(obj.get("stars", 0) or 0)
        useful_sum[bid] += int(obj.get("useful", 0) or 0)
        funny_sum[bid] += int(obj.get("funny", 0) or 0)
        cool_sum[bid] += int(obj.get("cool", 0) or 0)

        user_id = obj.get("user_id", "")
        if user_id:
            biz_user_ids[bid].append(user_id)
            all_user_ids.add(user_id)

        text = (obj.get("text") or "").strip().replace("\n", " ")
        if text and len(snippets[bid]) < max_snippets:
            snippets[bid].append(text[:320])

    rows = []
    for bid in target_ids:
        n = counts.get(bid, 0)
        rows.append(
            {
                "business_id": bid,
                "review_count_json": n,
                "avg_review_stars_json": round(star_sums[bid] / n, 2) if n else "",
                "review_useful_sum": useful_sum.get(bid, 0),
                "review_funny_sum": funny_sum.get(bid, 0),
                "review_cool_sum": cool_sum.get(bid, 0),
                "top_review_snippets": " || ".join(snippets.get(bid, [])),
            }
        )

    return pd.DataFrame(rows), biz_user_ids, all_user_ids


def aggregate_tips(path: Path, target_ids: set[str], max_snippets: int) -> pd.DataFrame:
    counts: Dict[str, int] = defaultdict(int)
    compliment_sum: Dict[str, int] = defaultdict(int)
    snippets: Dict[str, List[str]] = defaultdict(list)

    for obj in iter_json_lines(path):
        bid = obj.get("business_id")
        if bid not in target_ids:
            continue

        counts[bid] += 1
        # Yelp tip schema uses compliment_count.
        compliment_sum[bid] += int(obj.get("compliment_count", 0) or 0)

        text = (obj.get("text") or "").strip().replace("\n", " ")
        if text and len(snippets[bid]) < max_snippets:
            snippets[bid].append(text[:240])

    rows = []
    for bid in target_ids:
        rows.append(
            {
                "business_id": bid,
                "tip_count_json": counts.get(bid, 0),
                "tip_compliment_sum": compliment_sum.get(bid, 0),
                "top_tip_snippets": " || ".join(snippets.get(bid, [])),
            }
        )

    return pd.DataFrame(rows)


def aggregate_checkins(path: Path, target_ids: set[str]) -> pd.DataFrame:
    rows = []
    for obj in iter_json_lines(path):
        bid = obj.get("business_id")
        if bid not in target_ids:
            continue

        raw_dates = (obj.get("date") or "").strip()
        checkin_count = 0
        if raw_dates:
            checkin_count = len([x for x in raw_dates.split(",") if x.strip()])

        rows.append(
            {
                "business_id": bid,
                "checkin_count": checkin_count,
                "checkin_dates_raw": raw_dates,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["business_id", "checkin_count", "checkin_dates_raw"])
    return pd.DataFrame(rows)


def load_users(path: Path, keep_user_ids: set[str]) -> Dict[str, dict]:
    users: Dict[str, dict] = {}
    for obj in iter_json_lines(path):
        uid = obj.get("user_id")
        if uid not in keep_user_ids:
            continue
        users[uid] = {
            "review_count": int(obj.get("review_count", 0) or 0),
            "fans": int(obj.get("fans", 0) or 0),
            "average_stars": float(obj.get("average_stars", 0) or 0),
            "useful": int(obj.get("useful", 0) or 0),
            "funny": int(obj.get("funny", 0) or 0),
            "cool": int(obj.get("cool", 0) or 0),
        }
    return users


def aggregate_user_features_by_business(
    target_ids: set[str],
    biz_user_ids: Dict[str, List[str]],
    users: Dict[str, dict],
) -> pd.DataFrame:
    rows = []
    for bid in target_ids:
        user_ids = biz_user_ids.get(bid, [])
        if not user_ids:
            rows.append(
                {
                    "business_id": bid,
                    "distinct_reviewers": 0,
                    "avg_reviewer_stars": "",
                    "avg_reviewer_review_count": "",
                    "avg_reviewer_fans": "",
                }
            )
            continue

        uniq = list(dict.fromkeys(user_ids))
        matched = [users[uid] for uid in uniq if uid in users]
        if not matched:
            rows.append(
                {
                    "business_id": bid,
                    "distinct_reviewers": len(uniq),
                    "avg_reviewer_stars": "",
                    "avg_reviewer_review_count": "",
                    "avg_reviewer_fans": "",
                }
            )
            continue

        rows.append(
            {
                "business_id": bid,
                "distinct_reviewers": len(uniq),
                "avg_reviewer_stars": round(sum(x["average_stars"] for x in matched) / len(matched), 3),
                "avg_reviewer_review_count": round(sum(x["review_count"] for x in matched) / len(matched), 2),
                "avg_reviewer_fans": round(sum(x["fans"] for x in matched) / len(matched), 2),
            }
        )

    return pd.DataFrame(rows)


def build_embedding_text(row: pd.Series) -> str:
    parts = [
        f"Name: {row.get('name', '')}",
        (
            "Location: "
            f"{row.get('address', '')}, {row.get('city', '')}, "
            f"{row.get('state', '')} {row.get('postal_code', '')}"
        ),
        f"Categories: {row.get('categories', '')}",
        f"Business stars: {row.get('stars', '')}",
        f"Business review count: {row.get('review_count', '')}",
    ]

    if row.get("attributes", ""):
        parts.append(f"Attributes: {row.get('attributes')}")
    if row.get("hours", ""):
        parts.append(f"Hours: {row.get('hours')}")
    if row.get("avg_review_stars_json", "") != "":
        parts.append(f"Average review stars (from review.json): {row.get('avg_review_stars_json')}")
    if row.get("top_review_snippets", ""):
        parts.append(f"Representative reviews: {row.get('top_review_snippets')}")
    if row.get("top_tip_snippets", ""):
        parts.append(f"Tips: {row.get('top_tip_snippets')}")
    if row.get("checkin_count", "") != "":
        parts.append(f"Checkins: {row.get('checkin_count')}")

    return ". ".join(str(x).strip() for x in parts if str(x).strip())


def main() -> None:
    args = parse_args()

    business_path = Path(args.business).expanduser().resolve()
    review_path = Path(args.review).expanduser().resolve() if args.review else None
    tip_path = Path(args.tip).expanduser().resolve() if args.tip else None
    user_path = Path(args.user).expanduser().resolve() if args.user else None
    checkin_path = Path(args.checkin).expanduser().resolve() if args.checkin else None

    out_tabular = Path(args.out_tabular).expanduser().resolve()
    out_rag = Path(args.out_rag).expanduser().resolve()
    out_tabular.parent.mkdir(parents=True, exist_ok=True)
    out_rag.parent.mkdir(parents=True, exist_ok=True)

    businesses = load_businesses(business_path)
    target_ids = set(businesses["business_id"].tolist())
    merged = businesses

    biz_user_ids: Dict[str, List[str]] = {}
    all_user_ids: set[str] = set()

    if review_path and review_path.exists():
        review_df, biz_user_ids, all_user_ids = aggregate_reviews(
            review_path,
            target_ids,
            args.max_review_snippets,
        )
        merged = merged.merge(review_df, on="business_id", how="left")

    if tip_path and tip_path.exists():
        tip_df = aggregate_tips(tip_path, target_ids, args.max_tip_snippets)
        merged = merged.merge(tip_df, on="business_id", how="left")

    if checkin_path and checkin_path.exists():
        checkin_df = aggregate_checkins(checkin_path, target_ids)
        merged = merged.merge(checkin_df, on="business_id", how="left")

    if user_path and user_path.exists() and review_path and review_path.exists() and all_user_ids:
        users = load_users(user_path, all_user_ids)
        user_agg_df = aggregate_user_features_by_business(target_ids, biz_user_ids, users)
        merged = merged.merge(user_agg_df, on="business_id", how="left")

    merged = merged.fillna("")

    # Fields aligned with starter build script defaults.
    merged["neighborhood"] = merged["city"]
    merged["price"] = "N/A"
    merged["rating"] = merged["stars"]
    merged["highlights"] = (
        merged.get("top_tip_snippets", "").astype(str).str[:220]
        + " "
        + merged.get("top_review_snippets", "").astype(str).str[:300]
    ).str.strip()

    merged["embedding_text"] = merged.apply(build_embedding_text, axis=1)

    merged.sort_values(by=["review_count", "stars"], ascending=[False, False], inplace=True)
    merged.to_csv(out_tabular, index=False)

    rag_cols = [
        "business_id",
        "name",
        "address",
        "city",
        "state",
        "postal_code",
        "categories",
        "is_open",
        "hours",
        "price",
        "rating",
        "highlights",
        "embedding_text",
    ]
    rag_df = merged[[c for c in rag_cols if c in merged.columns]].copy()
    rag_df.to_csv(out_rag, index=False)

    print(f"Wrote tabular CSV: {out_tabular} ({len(merged)} rows)")
    print(f"Wrote RAG CSV: {out_rag} ({len(rag_df)} rows)")


if __name__ == "__main__":
    main()
