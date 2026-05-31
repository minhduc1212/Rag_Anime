import os
import chromadb
from google import genai
from google.genai import types
from sentence_transformers import SentenceTransformer
import dotenv

dotenv.load_dotenv()

# --- Configuration ---
EMBEDDING_MODEL = "BAAI/bge-large-en-v1.5"
BGE_QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

LLM_MODEL = "gemma-4-31b-it"  # As requested

def init_systems():
    print(f"Loading embedding model '{EMBEDDING_MODEL}'...")
    embedder = SentenceTransformer(EMBEDDING_MODEL)
    
    print("Connecting to ChromaDB...")
    chroma_client = chromadb.PersistentClient(path="./chroma_db_local")
    collection = chroma_client.get_collection(name="anime_collection")
    
    print(f"Initializing Google GenAI Client for {LLM_MODEL}...")
    # The client automatically picks up GEMINI_API_KEY from the environment
    genai_client = genai.Client()
    
    return embedder, collection, genai_client

def search_anime(query, embedder, collection, top_k=50):
    # 1. Prefix the query for BGE models
    full_query = BGE_QUERY_PREFIX + query
    
    # 2. Generate embedding for the query
    query_embedding = embedder.encode(full_query, normalize_embeddings=True).tolist()
    
    # 3. Search ChromaDB for the most similar vectors
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k
    )
    
    return results

def ask_llm(query, context_results, genai_client):
    # Format the context retrieved from ChromaDB
    context_text = ""
    if context_results['documents'] and len(context_results['documents']) > 0:
        docs = context_results['documents'][0]
        metadatas = context_results['metadatas'][0]
        
        # Combine documents and their metadata so we can sort them together
        combined_results = list(zip(docs, metadatas))
        
        # Sort the results by 'score' in descending order (highest score first)
        combined_results.sort(key=lambda x: float(x[1].get('score', 0.0)), reverse=True)
        
        for i, (doc, meta) in enumerate(combined_results):
            context_text += f"--- Anime {i+1} (Score: {meta.get('score', 0.0)}) ---\n"
            context_text += f"{doc}\n"
            context_text += f"URL: {meta.get('url', 'N/A')}\n\n"
    
    prompt = f"""You are a helpful, enthusiastic, and knowledgeable anime expert assistant.
Use the following retrieved context from our anime database to answer the user's question. 
The context contains up to 50 anime, sorted by their score from highest to lowest.
CRITICAL: You are acting as a data mapper. You MUST include EVERY SINGLE anime from the context in the `all_retrieved_anime` list below. Do not filter, truncate, or limit the list! If there are 50 anime in the context, you MUST output exactly 50 items in the `all_retrieved_anime` array.

Please return your response in strictly JSON format matching the following structure:
{{
  "message": "A string containing your overall response and explanation based on the user's question.",
  "all_retrieved_anime": [
    {{
      "title": "Anime Title",
      "url": "Anime URL"
    }}
  ]
}}

If the answer is not contained in the context, clearly state in the message that based on the provided search results, you don't have enough information, but you can try to provide a general answer. However, YOU MUST STILL LIST ALL RETRIEVED ANIME IN THE ARRAY.

Context from Database:
{context_text}

User Question: {query}
Answer:"""

    response = genai_client.models.generate_content(
        model=LLM_MODEL,
        contents=prompt,
        config=types.GenerateContentConfig(
            response_mime_type="application/json",
            max_output_tokens=16000,
            temperature=0.1
        )
    )
    
    return response.text

def main():
    try:
        embedder, collection, genai_client = init_systems()
    except Exception as e:
        print(f"Failed to initialize systems: {e}")
        return
        
    print("\n" + "="*60)
    print("Anime RAG System Ready!")
    print("="*60 + "\n")
    
    while True:
        query = input("\nWhat kind of anime are you looking for? (or 'quit'): ")
        if query.lower() in ['quit', 'exit', 'q']:
            print("Goodbye!")
            break
            
        if not query.strip():
            continue
            
        print("\n🔍 Searching vector database...")
        results = search_anime(query, embedder, collection, top_k=50)
        
        print(f"🧠 Generating answer with {LLM_MODEL}...")
        try:
            answer = ask_llm(query, results, genai_client)
            
            print("\n" + "-"*60)
            print("🤖 ANSWER:")
            print(answer)
            print("-"*60)
        except Exception as e:
            print(f"\n❌ Error calling LLM: {e}")

if __name__ == "__main__":
    main()
