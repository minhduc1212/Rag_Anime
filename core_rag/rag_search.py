"""
rag_search.py — Anime RAG Search (OpenAI embed + Gemini LLM)
=============================================================
Applies structured query rewrite, safe response parsing, and strict LLM ranking
from new_core.py while keeping the synchronous standalone architecture.
"""

import json
import os
import re
import logging
import dotenv
import chromadb
from dataclasses import dataclass
from openai import OpenAI
from google import genai
from google.genai import types

dotenv.load_dotenv()

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(levelname)s | %(message)s")


# ══════════════════════════════════════════════════════════════════════════════
#  Dataclasses
# ══════════════════════════════════════════════════════════════════════════════

@dataclass
class AnimeResult:
    rank: int
    title: str
    url: str
    mal_score: float
    why: str


@dataclass
class RetrievedItem:
    title: str
    url: str
    relevance: float


@dataclass
class SearchResponse:
    query: str
    rewritten_query: str
    excluded_titles: list[str]
    message: str
    recommendations: list[AnimeResult]
    all_retrieved: list[RetrievedItem]


# ══════════════════════════════════════════════════════════════════════════════
#  Config
# ══════════════════════════════════════════════════════════════════════════════

EMBED_MODEL    = "text-embedding-3-small"
DIMENSIONS     = 1536
LLM_MODEL      = "gemma-4-31b-it"
REWRITE_MODEL  = "gemma-4-31b-it"
CHROMA_PATH    = "./chroma_db"
COLLECTION     = "anime_collection"
TOP_K          = 20

openai_client = OpenAI()
genai_client  = genai.Client()


# ══════════════════════════════════════════════════════════════════════════════
#  Helpers (from new_core.py)
# ══════════════════════════════════════════════════════════════════════════════

def _safe_text(response) -> str | None:
    """
    Extract text từ Gemini response an toàn.
    response.text có thể raise hoặc trả None khi bị safety filter / MAX_TOKENS.
    """
    try:
        if response.text is not None:
            return response.text
    except Exception:
        pass
    try:
        for candidate in (response.candidates or []):
            for part in (candidate.content.parts or []):
                if hasattr(part, "text") and part.text:
                    return part.text
    except Exception:
        pass
    return None


def _parse_json_response(raw: str | None, fallback: dict) -> dict:
    """Parse JSON response, strip markdown fences, fallback nếu fail."""
    if not raw:
        return fallback
    raw = raw.strip()
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        log.warning("JSON decode failed, using fallback")
        return fallback


# ══════════════════════════════════════════════════════════════════════════════
#  Init
# ══════════════════════════════════════════════════════════════════════════════

def init_chromadb():
    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    return chroma.get_collection(name=COLLECTION)


# ══════════════════════════════════════════════════════════════════════════════
#  Embed query
# ══════════════════════════════════════════════════════════════════════════════

def embed_query(query: str) -> list[float]:
    """Embed query bằng text-embedding-3-small."""
    response = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=query,
        dimensions=DIMENSIONS,
    )
    return response.data[0].embedding


# ══════════════════════════════════════════════════════════════════════════════
#  Query Rewrite — STRUCTURED (from new_core.py)
# ══════════════════════════════════════════════════════════════════════════════

def rewrite_query(query: str) -> dict:
    """
    Dùng LLM để phân tích query thành các trường có cấu trúc, rồi build
    structured search string matching ChromaDB document format.

    Trả về:
      rewritten_query  — structured search string mirroring ChromaDB doc format
      excluded_titles  — anime cần loại khỏi kết quả
    """
    prompt = f"""You are an anime search assistant. Analyze the user query and extract structured fields.

User query: "{query}"

1. Extract any specific anime titles the user wants to find SIMILAR anime to -> excluded_titles.
2. Fill each field below. Be specific; use terms that appear in anime databases.

Respond ONLY with valid JSON, no markdown:
{{
  "excluded_titles": [],
  "genres": "comma-separated genres, e.g. Action, Adventure, Fantasy",
  "tags": "comma-separated tags, e.g. Isekai, Overpowered Protagonist, Magic System",
  "setting": "e.g. fantasy world, post-apocalyptic city, high school, outer space",
  "mood": "e.g. dark and gritty, wholesome, epic, comedic, melancholic",
  "themes": "e.g. redemption, friendship, war, survival, romance, coming-of-age",
  "plot_elements": "e.g. transported to another world, robot pilots, detective mystery, tournament arc",
  "similar_to": "comma-separated well-known anime titles with a similar feel (not the excluded ones)",
  "synopsis_keywords": "vivid descriptive phrases that would appear in a matching anime synopsis"
}}

Leave a field as empty string if not applicable."""

    try:
        response = genai_client.models.generate_content(
            model=REWRITE_MODEL,
            contents=[
                types.Content(role="user", parts=[types.Part(text=prompt)])
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
                max_output_tokens=600,
            ),
        )

        raw  = _safe_text(response)
        data = _parse_json_response(raw, {})

        excluded = data.get("excluded_titles") or []

        # Build a structured search string that mirrors the ChromaDB document format.
        # Documents are stored as:  genres: X \n tags: Y \n similar_to: Z \n synopsis: ...
        # Matching this structure puts the query embedding in the same vector space.
        parts = []
        if data.get("genres"):
            parts.append(f"genres: {data['genres']}")
        if data.get("tags"):
            parts.append(f"tags: {data['tags']}")
        if data.get("setting"):
            parts.append(f"setting: {data['setting']}")
        if data.get("mood"):
            parts.append(f"mood: {data['mood']}")
        if data.get("themes"):
            parts.append(f"themes: {data['themes']}")
        if data.get("plot_elements"):
            parts.append(f"plot_elements: {data['plot_elements']}")
        if data.get("similar_to"):
            parts.append(f"similar_to: {data['similar_to']}")
        if data.get("synopsis_keywords"):
            parts.append(f"synopsis: {data['synopsis_keywords']}")

        rewritten = "\n".join(parts) if parts else query
        log.info("Structured rewrite:\n%s", rewritten)

        return {
            "rewritten_query": rewritten,
            "excluded_titles": excluded,
        }
    except Exception as e:
        log.warning("Query rewrite failed: %s", e)
        return {"rewritten_query": query, "excluded_titles": []}


# ══════════════════════════════════════════════════════════════════════════════
#  Vector Search — with collection.count() cap (from new_core.py)
# ══════════════════════════════════════════════════════════════════════════════

def search(query: str, collection, top_k: int = TOP_K, excluded_titles: list[str] = None) -> list[dict]:
    """
    Tìm kiếm vector DB, trả về list anime đã được làm giàu metadata.
    Có lọc bỏ những anime có title trùng với excluded_titles.
    """
    embedding = embed_query(query)

    # Lấy dư 3× để bù cho những kết quả bị filter
    fetch_k = top_k * 3 if excluded_titles else top_k

    results = collection.query(
        query_embeddings=[embedding],
        n_results=min(fetch_k, collection.count()),
        include=["documents", "metadatas", "distances"],
    )

    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    excluded_lower = [t.lower() for t in (excluded_titles or [])]

    combined = []
    for doc, meta, dist in zip(docs, metadatas, distances):
        if excluded_lower:
            title = str(meta.get("title", "")).lower()
            if any(ex in title for ex in excluded_lower):
                continue
        combined.append({
            "doc":       doc,
            "meta":      meta,
            "relevance": round(1 - dist, 4),
        })

    # Sort: ưu tiên relevance trước, sau đó score MAL
    combined.sort(
        key=lambda x: (x["relevance"], float(x["meta"].get("score", 0))),
        reverse=True,
    )
    return combined[:top_k]


# ══════════════════════════════════════════════════════════════════════════════
#  Build context
# ══════════════════════════════════════════════════════════════════════════════

def build_context(results: list[dict]) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        m = r["meta"]
        lines.append(
            f"[{i}] {m.get('title', '?')} "
            f"(MAL score: {m.get('score', '?')} | relevance: {r['relevance']})\n"
            f"{r['doc']}\n"
            f"URL: {m.get('url', '')}\n"
        )
    return "\n".join(lines)


# ══════════════════════════════════════════════════════════════════════════════
#  LLM — strict prompt forcing ALL N ranked (from new_core.py)
# ══════════════════════════════════════════════════════════════════════════════

SYSTEM_PROMPT = (
    "You are an expert anime recommender. "
    "Analyze the retrieved list and answer accurately. "
    "Base answers ONLY on context. Rank by intent match, not score. "
    "Always respond in the EXACT JSON format. No text outside JSON."
)


def ask_llm(query: str, results: list[dict]) -> dict:
    context = build_context(results)
    n       = len(results)

    prompt = f"""Context ({n} anime retrieved, sorted by relevance):
{context}

User query: {query}

Respond ONLY with JSON:
{{
  "message": "Detailed answer explaining recommendations.",
  "recommendations": [
    {{
      "rank": 1,
      "title": "Anime Title",
      "url": "https://myanimelist.net/...",
      "mal_score": 8.5,
      "why": "One sentence why this matches."
    }}
  ],
  "all_retrieved": [
    {{"title": "Title", "url": "https://...", "relevance": 0.95}}
  ]
}}

recommendations: You MUST include EXACTLY {n} entries — rank ALL {n} retrieved anime from best to worst match. Do NOT skip or omit any. all_retrieved: ALL {n} anime in the same order."""

    fallback = {
        "message": "Could not generate answer.",
        "recommendations": [],
        "all_retrieved": [
            {"title": r["meta"].get("title", ""), "url": r["meta"].get("url", ""), "relevance": r["relevance"]}
            for r in results
        ],
    }

    try:
        response = genai_client.models.generate_content(
            model=LLM_MODEL,
            contents=[
                types.Content(
                    role="user",
                    parts=[types.Part(text=SYSTEM_PROMPT + "\n\n" + prompt)]
                )
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                max_output_tokens=8192,
                temperature=0.1,
            ),
        )

        raw = _safe_text(response)
        if not raw:
            try:
                reason = response.candidates[0].finish_reason
                log.warning("LLM blocked: finish_reason=%s", reason)
            except Exception:
                pass
            return fallback

        parsed = _parse_json_response(raw, fallback)

        # Save raw result for debugging
        try:
            with open("result.json", "w", encoding="utf-8") as f:
                json.dump({"raw": raw}, f, indent=2, ensure_ascii=False)
        except Exception:
            pass

        return parsed

    except Exception as e:
        log.error("LLM error: %s", e)
        return fallback


# ══════════════════════════════════════════════════════════════════════════════
#  Display
# ══════════════════════════════════════════════════════════════════════════════

def display(answer: dict):
    print("\n" + "═" * 60)
    print("💬 " + answer.get("message", ""))

    recs = answer.get("recommendations", [])
    if recs:
        print(f"\n🎯 Top {len(recs)} Recommendations:")
        for r in recs:
            print(f"  {r.get('rank', '?')}. {r.get('title')}  (score: {r.get('mal_score', '?')})")
            print(f"     {r.get('why', '')}")
            print(f"     🔗 {r.get('url', '')}")

    all_r = answer.get("all_retrieved", [])
    if all_r:
        print(f"\n📋 All {len(all_r)} retrieved results:")
        for r in all_r:
            print(f"  • {r.get('title')}  (relevance: {r.get('relevance', '?')})  {r.get('url', '')}")

    print("═" * 60)


# ══════════════════════════════════════════════════════════════════════════════
#  Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("Connecting to ChromaDB…")
    try:
        collection = init_chromadb()
        print(f"✅ Collection loaded: {collection.count()} docs\n")
    except Exception as e:
        print(f"❌ ChromaDB error: {e}")
        return

    print("═" * 60)
    print("  🎌 Anime RAG Search  (OpenAI embed + Gemma 4)")
    print("═" * 60)

    while True:
        try:
            query = input("\n🔍 Query (or 'q' to quit): ").strip()
        except (KeyboardInterrupt, EOFError):
            break

        if not query or query.lower() in ("q", "quit", "exit"):
            print("Goodbye!")
            break

        try:
            print(f"   Rewriting query using {REWRITE_MODEL}…")
            rewrite_res = rewrite_query(query)
            rewritten_q = rewrite_res.get("rewritten_query", query)
            excluded_titles = rewrite_res.get("excluded_titles", [])

            if excluded_titles:
                print(f"   Excluded titles: {', '.join(excluded_titles)}")
            print(f"   Rewritten query: {rewritten_q}")

            print("   Embedding query…")
            results = search(rewritten_q, collection, top_k=TOP_K, excluded_titles=excluded_titles)
            print(f"   Found {len(results)} candidates. Asking {LLM_MODEL}…")
            answer  = ask_llm(query, results)
            display(answer)
        except Exception as e:
            log.error("Error: %s", e)


if __name__ == "__main__":
    main()