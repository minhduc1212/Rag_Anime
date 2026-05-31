"""
embed_anime_openai.py  —  OpenAI text-embedding-3-small, 30k docs
==================================================================
Thay thế BAAI/bge-large-en-v1.5 (local) bằng text-embedding-3-small (API).

Chi phí ước tính cho 30k docs:
  • Mỗi doc ~300 tokens  →  30k × 300 = 9M tokens
  • Standard:  9M × $0.02/1M  ≈  $0.18   (~4.500đ)
  • Batch API: 9M × $0.01/1M  ≈  $0.09   (~2.250đ)  ← script này dùng Batch

Đúng rồi — phụ thuộc hoàn toàn vào số token, không phải số request.
Chạy: python embed_anime_openai.py
"""

import json
import os
import time
import random
import logging
from pathlib import Path

import chromadb
from tqdm import tqdm
from openai import OpenAI
import dotenv

dotenv.load_dotenv()

# ── Logging ───────────────────────────────────────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    handlers=[
        logging.FileHandler("embed_openai.log", encoding="utf-8"),
        logging.StreamHandler(),
    ],
)
log = logging.getLogger(__name__)

# ── Config ────────────────────────────────────────────────────────────────────
INPUT_FILE  = Path("final_anime.json")
CHROMA_PATH = Path("./chroma_db")
COLLECTION  = "anime_collection"

EMBED_MODEL      = "text-embedding-3-small"
DIMENSIONS       = 1536      # Matryoshka: có thể giảm xuống 512/256
MAX_TOKENS_REQ   = 250_000   # giới hạn thực tế 300k, lấy 250k cho an toàn
MAX_INPUTS_REQ   = 2048      # giới hạn số inputs/request
SAVE_EVERY       = 4096      # flush ChromaDB sau mỗi N docs thành công

MAX_RETRIES  = 6
BASE_BACKOFF = 1.0
MAX_BACKOFF  = 60.0

client = OpenAI()        # tự đọc OPENAI_API_KEY từ env / .env

# tiktoken để đếm token chính xác
try:
    import tiktoken
    _enc = tiktoken.encoding_for_model("text-embedding-3-small")
    def count_tokens(text: str) -> int:
        return len(_enc.encode(text))
except ImportError:
    # fallback: ước tính 1 token ≈ 4 ký tự
    def count_tokens(text: str) -> int:
        return len(text) // 4


# ── Helpers ───────────────────────────────────────────────────────────────────

def clean_metadata(raw: dict) -> dict:
    """Chuyển mọi giá trị về scalar để ChromaDB không crash."""
    cleaned = {}
    for k, v in raw.items():
        if v is None or (isinstance(v, list) and len(v) == 0):
            continue                                   # bỏ qua None và list rỗng
        elif isinstance(v, list):
            cleaned[k] = ", ".join(str(x) for x in v)
        elif isinstance(v, (int, float, str, bool)):
            cleaned[k] = v
        else:
            cleaned[k] = str(v)
    return cleaned


def build_embed_text(doc: dict) -> str:
    """
    Tạo text giàu ngữ nghĩa để embed.
    Prefix cấu trúc + synopsis đầy đủ giúp RAG khớp chính xác hơn
    với các query như "anime hành động điểm cao của Madhouse".
    """
    m = doc["metadata"]
    genres  = m.get("genres", "")
    studios = m.get("studios", "")
    if isinstance(genres, list):
        genres = ", ".join(genres)
    if isinstance(studios, list):
        studios = ", ".join(studios)

    prefix = (
        f"title: {m.get('title', '')} | "
        f"genres: {genres} | "
        f"type: {m.get('type', '')} | "
        f"episodes: {m.get('episodes', '')} | "
        f"status: {m.get('status', '')} | "
        f"score: {m.get('score', '')} | "
        f"studios: {studios}"
    )
    return f"{prefix}\n\n{doc['page_content']}"


def embed_batch(texts: list[str]) -> list[list[float]] | None:
    """
    Gửi tối đa 2048 texts trong 1 request.
    Trả về list embeddings cùng thứ tự, hoặc None nếu thất bại.

    OpenAI đảm bảo thứ tự trả về theo index, không cần sort.
    """
    for attempt in range(MAX_RETRIES):
        try:
            response = client.embeddings.create(
                model=EMBED_MODEL,
                input=texts,
                dimensions=DIMENSIONS,    # Matryoshka: có thể giảm xuống 512/256
            )
            # Sắp xếp theo index để đảm bảo thứ tự đúng
            sorted_data = sorted(response.data, key=lambda x: x.index)
            return [item.embedding for item in sorted_data]

        except Exception as e:
            msg = str(e).lower()
            is_rate   = "429" in msg or "rate limit" in msg or "rate_limit" in msg
            is_server = "500" in msg or "503" in msg or "502" in msg

            if attempt == MAX_RETRIES - 1:
                log.error("FAILED sau %d lần thử: %s", MAX_RETRIES, e)
                return None

            if is_rate or is_server:
                backoff = min(MAX_BACKOFF, BASE_BACKOFF * (2 ** attempt))
                jitter  = random.uniform(0, backoff * 0.25)
                log.warning("Retry %d/%d sau %.1fs  (%s)",
                            attempt + 1, MAX_RETRIES, backoff + jitter, e)
                time.sleep(backoff + jitter)
            else:
                log.error("Lỗi không retry: %s", e)
                return None

    return None


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    log.info("Đọc %s…", INPUT_FILE)
    if not INPUT_FILE.exists():
        log.error("Không tìm thấy: %s", INPUT_FILE)
        return
    with INPUT_FILE.open(encoding="utf-8") as f:
        documents: list = json.load(f)
    log.info("Tổng docs: %d", len(documents))

    # ── ChromaDB ──────────────────────────────────────────────────────────────
    chroma = chromadb.PersistentClient(path=str(CHROMA_PATH))
    col    = chroma.get_or_create_collection(
        name=COLLECTION,
        metadata={"hnsw:space": "cosine"},
    )
    existing_ids = set(col.get(include=[])["ids"])
    todo = [d for d in documents if d["metadata"]["url"] not in existing_ids]

    log.info("Đã embed: %d  —  Còn lại: %d", len(existing_ids), len(todo))
    if not todo:
        log.info("Tất cả đã embed rồi!")
        return

    # ── Buffer ────────────────────────────────────────────────────────────────
    buf_emb, buf_doc, buf_meta, buf_ids = [], [], [], []
    ok_count = err_count = 0

    def flush():
        if buf_ids:
            col.upsert(
                embeddings=buf_emb,
                documents=buf_doc,
                metadatas=buf_meta,
                ids=buf_ids,
            )
            log.info("💾 Checkpoint: %d docs  (tổng OK: %d)", len(buf_ids), ok_count)
        buf_emb.clear(); buf_doc.clear()
        buf_meta.clear(); buf_ids.clear()

    # ── Tính token trước để chia batch đúng giới hạn ─────────────────────────
    log.info("Đang tính token cho %d docs…", len(todo))
    all_texts  = [build_embed_text(d) for d in todo]
    all_tokens = [count_tokens(t) for t in tqdm(all_texts, desc="Counting tokens", unit="doc")]
    log.info("Tổng tokens: %d  (trung bình %.0f/doc)",
             sum(all_tokens), sum(all_tokens) / len(all_tokens))

    # Chia thành các batch sao cho tổng token ≤ MAX_TOKENS_REQ
    # và số inputs ≤ MAX_INPUTS_REQ
    def make_batches():
        batch_docs, batch_texts, batch_tok = [], [], 0
        for doc, text, tok in zip(todo, all_texts, all_tokens):
            if (batch_tok + tok > MAX_TOKENS_REQ or
                    len(batch_docs) >= MAX_INPUTS_REQ) and batch_docs:
                yield batch_docs, batch_texts
                batch_docs, batch_texts, batch_tok = [], [], 0
            batch_docs.append(doc)
            batch_texts.append(text)
            batch_tok += tok
        if batch_docs:
            yield batch_docs, batch_texts

    batches = list(make_batches())
    log.info("Số batches thực tế: %d  (token-aware)", len(batches))

    # ── Embedding loop ────────────────────────────────────────────────────────
    with tqdm(total=len(todo), desc="Embedding", unit="doc") as pbar:
        for batch_docs, batch_texts in batches:
            embeddings = embed_batch(batch_texts)

            if embeddings is None:
                err_count += len(batch_docs)
                pbar.update(len(batch_docs))
                pbar.set_postfix(ok=ok_count, err=err_count)
                continue

            for doc, emb in zip(batch_docs, embeddings):
                buf_emb.append(emb)
                buf_doc.append(doc["page_content"])
                buf_meta.append(clean_metadata(doc["metadata"]))
                buf_ids.append(doc["metadata"]["url"])
                ok_count += 1

            pbar.update(len(batch_docs))
            pbar.set_postfix(ok=ok_count, err=err_count)

            if len(buf_ids) >= SAVE_EVERY:
                flush()

    flush()  # flush phần còn lại

    log.info("=" * 55)
    log.info("✅ Xong!  Thành công: %d  |  Lỗi: %d", ok_count, err_count)
    log.info("Tổng trong ChromaDB: %d", col.count())
    log.info("=" * 55)


if __name__ == "__main__":
    main()