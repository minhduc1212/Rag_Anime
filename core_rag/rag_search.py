import json
import os
import re
import dotenv
import chromadb
from openai import OpenAI
from google import genai
from google.genai import types

dotenv.load_dotenv()

# ── Config ────────────────────────────────────────────────────────────────────
EMBED_MODEL  = "text-embedding-3-small"
DIMENSIONS   = 1536
LLM_MODEL    = "gemma-4-31b-it"
REWRITE_MODEL = "gemma-4-31b-it"
CHROMA_PATH  = "./chroma_db"
COLLECTION   = "anime_collection"
TOP_K        = 20

openai_client = OpenAI()
genai_client  = genai.Client()


# ── Init ──────────────────────────────────────────────────────────────────────

def init_chromadb():
    chroma = chromadb.PersistentClient(path=CHROMA_PATH)
    return chroma.get_collection(name=COLLECTION)


# ── Embed query ───────────────────────────────────────────────────────────────

def embed_query(query: str) -> list[float]:
    """
    Embed query bằng text-embedding-3-small.
    Dùng RETRIEVAL_QUERY task type để tối ưu độ chính xác tìm kiếm.
    """
    response = openai_client.embeddings.create(
        model=EMBED_MODEL,
        input=query,
        dimensions=DIMENSIONS,
    )
    return response.data[0].embedding


# ── Query Rewrite ─────────────────────────────────────────────────────────────

def rewrite_query(query: str) -> dict:
    """
    Dùng LLM để viết lại query cho search và trích xuất các title cần loại trừ.
    """
    prompt = f"""You are an anime search assistant.
User query: {query}

Tasks:
1. Extract any specific anime titles mentioned in the query that the user wants to find similar anime to.
2. Rewrite the query to be an optimal semantic search query for a vector database (describe the genres, themes, plot elements, tropes). Do NOT include the extracted titles in the rewritten query.

Respond ONLY with this JSON structure (no markdown, no extra text):
{{
  "rewritten_query": "The rewritten semantic search query",
  "excluded_titles": ["title1", "title2"]
}}"""

    try:
        response = genai_client.models.generate_content(
            model=REWRITE_MODEL,
            contents=[
                types.Content(role="user", parts=[types.Part(text=prompt)])
            ],
            config=types.GenerateContentConfig(
                response_mime_type="application/json",
                temperature=0.1,
            ),
        )

        raw = response.text.strip()
        raw = re.sub(r"^```(?:json)?\s*", "", raw)
        raw = re.sub(r"\s*```$", "", raw)

        data = json.loads(raw)
        return {
            "rewritten_query": data.get("rewritten_query", query),
            "excluded_titles": data.get("excluded_titles", [])
        }
    except Exception as e:
        print(f"⚠️ Query rewrite failed: {e}")
        return {"rewritten_query": query, "excluded_titles": []}


# ── Search ────────────────────────────────────────────────────────────────────

def search(query: str, collection, top_k: int = TOP_K, excluded_titles: list[str] = None) -> list[dict]:
    """
    Tìm kiếm vector DB, trả về list anime đã được làm giàu metadata.
    Có lọc bỏ những anime có title trùng với excluded_titles.
    """
    embedding = embed_query(query)

    # Lấy dư ra để bù vào những kết quả bị filter
    fetch_k = top_k * 3 if excluded_titles else top_k

    results = collection.query(
        query_embeddings=[embedding],
        n_results=fetch_k,
        include=["documents", "metadatas", "distances"],
    )

    docs      = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]   # cosine distance (nhỏ hơn = gần hơn)

    # Kết hợp + sắp xếp theo relevance score (1 - distance)
    combined = []
    for doc, meta, dist in zip(docs, metadatas, distances):
        # Metadata Filtering (ILIKE '%title%')
        if excluded_titles:
            title = str(meta.get("title", "")).lower()
            skip = False
            for ex_title in excluded_titles:
                if ex_title.lower() in title:
                    skip = True
                    break
            if skip:
                continue

        combined.append({
            "doc":       doc,
            "meta":      meta,
            "relevance": round(1 - dist, 4),   # 0–1, càng cao càng liên quan
        })

    # Sort: ưu tiên relevance trước, sau đó score MAL
    combined.sort(key=lambda x: (x["relevance"], float(x["meta"].get("score", 0))), reverse=True)
    
    # Trả về đúng số lượng top_k yêu cầu
    return combined[:top_k]


# ── Build context ─────────────────────────────────────────────────────────────

def build_context(results: list[dict]) -> str:
    lines = []
    for i, r in enumerate(results, 1):
        m = r["meta"]
        lines.append(
            f"[{i}] {m.get('title', '?')}  "
            f"(MAL score: {m.get('score', '?')} | relevance: {r['relevance']})\n"
            f"{r['doc']}\n"
            f"URL: {m.get('url', '')}\n"
        )
    return "\n".join(lines)


# ── LLM ───────────────────────────────────────────────────────────────────────

SYSTEM_PROMPT = """You are an expert anime recommender assistant.
Your job is to analyze the retrieved anime list and answer the user's question accurately.

Rules:
- Base your answer ONLY on the provided context.
- Rank recommendations by how well they match the user's intent, NOT just by MAL score.
- Be specific about WHY each anime matches the query.
- If the context lacks relevant results, say so honestly.
- Always respond in the EXACT JSON format specified. No extra text outside JSON.
"""

def ask_llm(query: str, results: list[dict]) -> dict:
    context = build_context(results)

    prompt = f"""Context (retrieved anime, sorted by relevance):
{context}

User query: {query}

Respond ONLY with this JSON structure (no markdown, no extra text):
{{
  "message": "Your detailed answer explaining recommendations and why they match the query.",
  "recommendations": [
    {{
      "rank": 1,
      "title": "Anime Title",
      "url": "https://myanimelist.net/...",
      "mal_score": 8.5,
      "why": "One sentence explaining why this matches the query."
    }}
  ],
  "all_retrieved": [
    {{"title": "Title", "url": "https://...", "relevance": 0.95}}
  ]
}}

- `recommendations`: top picks that best match the query (max 10, ordered by fit).
- `all_retrieved`: ALL {len(results)} anime from context, in order received.
"""

    response = genai_client.models.generate_content(
        model=LLM_MODEL,
        contents=[
            types.Content(role="user", parts=[types.Part(text=SYSTEM_PROMPT + "\n\n" + prompt)])
        ],
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            max_output_tokens=8192,
            temperature=0.1,
        ),
    )

    raw = response.text.strip()

    # Strip markdown code fences nếu model trả về ```json ... ```
    raw = re.sub(r"^```(?:json)?\s*", "", raw)
    raw = re.sub(r"\s*```$", "", raw)

    try:
        with open("result.json", "w", encoding="utf-8") as f:
            json.dump({"raw": raw}, f, indent=2, ensure_ascii=False)
        return json.loads(raw)
    except json.JSONDecodeError:
        # Fallback: trả nguyên text trong message
        return {
            "message": raw,
            "recommendations": [],
            "all_retrieved": [
                {"title": r["meta"].get("title", "?"), "url": r["meta"].get("url", ""), "relevance": r["relevance"]}
                for r in results
            ],
        }


# ── Display ───────────────────────────────────────────────────────────────────

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


# ── Main ──────────────────────────────────────────────────────────────────────

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
            print(f"❌ Error: {e}")


if __name__ == "__main__":
    main()