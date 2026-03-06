from __future__ import annotations

import json
from datetime import datetime
from typing import List, Optional

from langchain_chroma import Chroma
from langchain_core.documents import Document
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from app.config import Settings


class PlaceRAG:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.embedding = OpenAIEmbeddings(
            api_key=settings.openai_api_key,
            model=settings.embed_model,
        )
        self.vectorstore = Chroma(
            persist_directory=settings.chroma_dir,
            embedding_function=self.embedding,
        )
        self.llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            model_name=settings.chat_model,
            temperature=0.4,
        )
        self.prompt = ChatPromptTemplate.from_template(
            """You recommend places to eat and drink.
Use the provided context and apply any explicit filters.
If context is sparse, say so briefly and still return best matches.

User request:
{query}

Filters:
Location: {location}
Day: {day}
Time: {time}

Candidate places:
{context}

Return:
1) Top 3 recommendations
2) For each: name, location, open-hours relevance, and why it matches
3) A short caveat if hours/location data is incomplete
"""
        )

    def recommend(
        self,
        query: str,
        location: str = "",
        day: str = "",
        time_str: str = "",
    ) -> str:
        docs = self.vectorstore.similarity_search(query, k=max(self.settings.top_k * 15, 60))
        filtered = self._filter_docs(docs, location=location, day=day, time_str=time_str)

        selected = filtered[: self.settings.top_k] if filtered else docs[: self.settings.top_k]
        context = self._format_docs(selected)

        response = self.llm.invoke(
            self.prompt.format_messages(
                query=query,
                location=location or "none",
                day=day or "none",
                time=time_str or "none",
                context=context,
            )
        )
        return response.content

    def _filter_docs(
        self,
        docs: List[Document],
        location: str,
        day: str,
        time_str: str,
    ) -> List[Document]:
        out: List[Document] = []
        loc = location.strip().lower()
        normalized_day = self._normalize_day(day)

        for doc in docs:
            md = doc.metadata or {}
            if not self._matches_location(md, loc):
                continue
            if not self._matches_hours(md, normalized_day, time_str):
                continue
            out.append(doc)
        return out

    @staticmethod
    def _matches_location(metadata: dict, location_lower: str) -> bool:
        if not location_lower:
            return True
        hay = " ".join(
            [
                str(metadata.get("address", "")),
                str(metadata.get("city", "")),
                str(metadata.get("state", "")),
            ]
        ).lower()
        return location_lower in hay

    def _matches_hours(self, metadata: dict, day: Optional[str], time_str: str) -> bool:
        is_open_flag = str(metadata.get("is_open", "")).strip()
        if is_open_flag == "0":
            return False

        if not day and not time_str:
            return True

        hours_raw = str(metadata.get("hours", "") or "").strip()
        if not hours_raw:
            return True

        hours = self._parse_hours_json(hours_raw)
        if not hours:
            return True

        if day and day not in hours:
            return False

        if not day:
            return True

        if not time_str:
            return bool(hours.get(day))

        return self._time_in_day_ranges(time_str, hours.get(day, ""))

    @staticmethod
    def _parse_hours_json(hours_raw: str) -> dict:
        try:
            obj = json.loads(hours_raw)
            return obj if isinstance(obj, dict) else {}
        except Exception:
            return {}

    @staticmethod
    def _normalize_day(day: str) -> Optional[str]:
        if not day:
            return None
        mapping = {
            "mon": "Monday",
            "monday": "Monday",
            "tue": "Tuesday",
            "tues": "Tuesday",
            "tuesday": "Tuesday",
            "wed": "Wednesday",
            "wednesday": "Wednesday",
            "thu": "Thursday",
            "thurs": "Thursday",
            "thursday": "Thursday",
            "fri": "Friday",
            "friday": "Friday",
            "sat": "Saturday",
            "saturday": "Saturday",
            "sun": "Sunday",
            "sunday": "Sunday",
        }
        return mapping.get(day.strip().lower())

    @staticmethod
    def _to_minutes(time_str: str) -> Optional[int]:
        try:
            t = datetime.strptime(time_str.strip(), "%H:%M")
            return t.hour * 60 + t.minute
        except Exception:
            return None

    def _time_in_day_ranges(self, target_time: str, day_range: str) -> bool:
        target = self._to_minutes(target_time)
        if target is None:
            return True

        if not day_range:
            return False

        ranges = [x.strip() for x in day_range.split(",") if x.strip()]
        for block in ranges:
            if "-" not in block:
                continue
            start_raw, end_raw = [x.strip() for x in block.split("-", 1)]
            start = self._to_minutes(start_raw)
            end = self._to_minutes(end_raw)
            if start is None or end is None:
                continue

            if end >= start and start <= target <= end:
                return True
            # Overnight window, e.g. 18:00-02:00
            if end < start and (target >= start or target <= end):
                return True

        return False

    @staticmethod
    def _format_docs(docs: List[Document]) -> str:
        lines = []
        for i, doc in enumerate(docs, start=1):
            md = doc.metadata or {}
            name = md.get("name", "Unknown place")
            address = md.get("address", "Address not available")
            city = md.get("city", "")
            state = md.get("state", "")
            categories = md.get("categories", "")
            hours = md.get("hours", "")
            lines.append(
                f"{i}. {name} | {address} {city} {state} | Categories: {categories} | Hours: {hours} | {doc.page_content}"
            )
        return "\n".join(lines)
