from pathlib import Path
import shutil
import sys

import pandas as pd
from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings


ROOT = Path(__file__).resolve().parents[1]
DEFAULT_DATA_FILE = ROOT / "data" / "cafes.csv"
DEFAULT_BATCH_SIZE = 128
DEFAULT_MAX_CHARS = 1800

# Allow direct script execution: `python scripts/build_vectorstore.py`
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.config import get_settings


def make_doc_text(row: pd.Series, max_chars: int) -> str:
    if "embedding_text" in row and str(row.get("embedding_text", "")).strip():
        text = str(row["embedding_text"])
    else:
        text = (
            f"{row.get('name', '')}. "
            f"Neighborhood: {row.get('neighborhood', row.get('city', ''))}. "
            f"Category: {row.get('category', row.get('categories', ''))}. "
            f"Price: {row.get('price', 'N/A')}. "
            f"Rating: {row.get('rating', row.get('stars', ''))}/5. "
            f"Highlights: {row.get('highlights', '')}."
        )

    text = " ".join(text.split())
    return text[:max_chars]


def resolve_metadata(row: pd.Series) -> dict:
    address = str(row.get("address", "") or "").strip()
    city = str(row.get("city", "") or "").strip()
    state = str(row.get("state", "") or "").strip()
    postal_code = str(row.get("postal_code", "") or "").strip()

    if not address:
        address = ", ".join([x for x in [city, state, postal_code] if x]).strip(", ")

    return {
        "name": str(row.get("name", "") or ""),
        "address": address or "Address not provided",
        "business_id": str(row.get("business_id", "") or ""),
        "city": city,
        "state": state,
        "categories": str(row.get("categories", "") or ""),
        "hours": str(row.get("hours", "") or ""),
        "is_open": str(row.get("is_open", "") or ""),
    }


def parse_runtime_args(argv: list[str]) -> tuple[Path, int, int, bool]:
    data_file = DEFAULT_DATA_FILE
    batch_size = DEFAULT_BATCH_SIZE
    max_chars = DEFAULT_MAX_CHARS
    reset = True

    for arg in argv:
        if arg.startswith("--batch-size="):
            batch_size = int(arg.split("=", 1)[1])
        elif arg.startswith("--max-chars="):
            max_chars = int(arg.split("=", 1)[1])
        elif arg == "--append":
            reset = False
        elif arg.startswith("--"):
            raise ValueError(f"Unknown option: {arg}")
        else:
            data_file = Path(arg).resolve()

    if batch_size <= 0:
        raise ValueError("batch size must be positive")
    if max_chars <= 0:
        raise ValueError("max chars must be positive")

    return data_file, batch_size, max_chars, reset


def chunked(items: list[Document], size: int):
    for i in range(0, len(items), size):
        yield items[i : i + size]


def main() -> None:
    settings = get_settings()
    data_file, batch_size, max_chars, reset = parse_runtime_args(sys.argv[1:])

    if not data_file.exists():
        raise FileNotFoundError(f"Data file not found: {data_file}")

    df = pd.read_csv(data_file)

    docs = []
    for _, row in df.iterrows():
        docs.append(
            Document(
                page_content=make_doc_text(row, max_chars=max_chars),
                metadata=resolve_metadata(row),
            )
        )

    embeddings = OpenAIEmbeddings(
        api_key=settings.openai_api_key,
        model=settings.embed_model,
    )

    persist_dir = ROOT / settings.chroma_dir
    if reset and persist_dir.exists():
        shutil.rmtree(persist_dir)

    vectorstore = Chroma(
        persist_directory=str(persist_dir),
        embedding_function=embeddings,
    )

    for idx, batch in enumerate(chunked(docs, batch_size), start=1):
        vectorstore.add_documents(batch)
        print(f"Embedded batch {idx}: {len(batch)} records")

    print(f"Built vector store with {len(docs)} records from {data_file}")
    print(f"Persisted at {persist_dir}")


if __name__ == "__main__":
    main()
