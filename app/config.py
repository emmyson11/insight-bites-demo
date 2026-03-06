import os
from dataclasses import dataclass
from dotenv import load_dotenv


load_dotenv()


@dataclass(frozen=True)
class Settings:
    openai_api_key: str
    embed_model: str = "text-embedding-3-small"
    chat_model: str = "gpt-4o-mini"
    top_k: int = 3
    chroma_dir: str = "chroma_db"



def get_settings() -> Settings:
    key = os.getenv("OPENAI_API_KEY", "").strip()
    if not key:
        raise ValueError("Missing OPENAI_API_KEY. Add it to your .env file.")

    return Settings(
        openai_api_key=key,
        embed_model=os.getenv("EMBED_MODEL", "text-embedding-3-small"),
        chat_model=os.getenv("CHAT_MODEL", "gpt-4o-mini"),
        top_k=int(os.getenv("TOP_K", "3")),
        chroma_dir=os.getenv("CHROMA_DIR", "chroma_db"),
    )
