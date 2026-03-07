# Insight Bites Starter App

A simple prototype of Insight Bites that use a Retrieval-Augmented Generation (RAG) for a eat-and-drink recommendation app

Try out the demo version here: https://insight-bites-demo.onrender.com/

## What this repo includes

- Flask backend with two routes (`/` and `/recommend`)
- LangChain + Chroma vector search
- OpenAI embeddings and chat model
- Simple web UI in plain HTML/CSS/JS
- Script to build your vector database
- Small sample dataset so you can run quickly

## Project structure

```
insight-bites-demo/
  app/
    config.py
    rag.py
    server.py
  data/
    cafes.csv
  scripts/
    build_vectorstore.py
  templates/
    index.html
  .env.example
  requirements.txt
```

## 1) Setup

```bash
cd path_to_repo
/opt/homebrew/bin/python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## 2) Configure API key

Create `.env` from example:

```bash
cp .env.example .env
```

Then edit `.env` and set:

```env
OPENAI_API_KEY=your_openai_api_key
```

## 3) Build the vector store

```bash
python scripts/build_vectorstore.py
```

This creates `starter_app/chroma_db/`.

## 4) Run the app

```bash
python -m app.server
```

Open [http://127.0.0.1:5000](http://127.0.0.1:5000).

## How requests flow

1. User enters preferences in UI.
2. Frontend calls `POST /recommend`.
3. Backend retrieves top matching places from Chroma.
4. LLM formats recommendations.
5. Response appears in browser.

## Using Large Yelp JSON Datasets

If your source files are Yelp JSONL (one JSON object per line), use:

```bash
python scripts/prepare_yelp_rag_csv.py \
  --business /absolute/path/yelp_academic_dataset_business.json \
  --review /absolute/path/yelp_academic_dataset_review.json \
  --tip /absolute/path/yelp_academic_dataset_tip.json \
  --user /absolute/path/yelp_academic_dataset_user.json \
  --checkin /absolute/path/yelp_academic_dataset_checkin.json \
  --out-tabular /Users/emmyson/Insight-Coffee/starter_app/data/yelp_places_tabular.csv \
  --out-rag /Users/emmyson/Insight-Coffee/starter_app/data/yelp_places_for_rag.csv
```

Then build embeddings from the generated RAG CSV:

```bash
python scripts/build_vectorstore.py /Users/emmyson/Insight-Coffee/starter_app/data/yelp_places_for_rag.csv
```

The converter script filters to eat/drink categories (restaurants, cafes, bars, bakeries, and related food spots) and creates an `embedding_text` column ready for vector embeddings.


## Demo Mode (No API Key Needed)

Use this when you want to share the app without exposing or spending your API token.

1. Set `DEMO_MODE=true` in `.env` (you can leave `OPENAI_API_KEY` blank).
2. Run the server normally.

```bash
cd /Users/emmyson/Insight-Coffee/starter_app
source .venv/bin/activate
DEMO_MODE=true python -m app.server
```

In demo mode, `/recommend` returns local sample responses from `data/demo_responses.json` and does not call OpenAI.
