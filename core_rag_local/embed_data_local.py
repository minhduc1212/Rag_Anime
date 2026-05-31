import json
import os
import chromadb
from tqdm import tqdm
from sentence_transformers import SentenceTransformer

MODEL_NAME = "BAAI/bge-large-en-v1.5"

print(f"Loading local embedding model '{MODEL_NAME}'... (This might take a moment to download if it's the first time)")
model = SentenceTransformer(MODEL_NAME)

BATCH_SIZE = 128  

def main():
    print("Loading prepared data...")
    try:
        with open('cleaned_anime_docs.json', 'r', encoding='utf-8') as f:
            documents = json.load(f)
    except FileNotFoundError:
        print("Error: cleaned_anime_docs.json not found. Run rag_prepare.py first.")
        return
        
    print(f"Total documents available: {len(documents)}")

    # Initialize ChromaDB persistent client
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_or_create_collection(
        name="anime_collection",
        metadata={"hnsw:space": "cosine"} # BGE models work best with cosine similarity
    )
    
    # 1. Retrieve existing IDs to resume if interrupted
    existing_data = collection.get(include=[])
    existing_ids = set(existing_data['ids'])
    print(f"Documents already embedded in ChromaDB: {len(existing_ids)}")
    
    # 2. Filter out documents that are already embedded
    docs_to_process = []
    for doc in documents:
        # Using URL as the unique document ID
        doc_id = doc["metadata"]["url"]
        if doc_id not in existing_ids:
            docs_to_process.append(doc)
            
    print(f"Documents left to process: {len(docs_to_process)}")
    
    if not docs_to_process:
        print("All documents have been embedded!")
        return

    total_batches = (len(docs_to_process) + BATCH_SIZE - 1) // BATCH_SIZE
    print(f"Starting local embedding process in {total_batches} batches...")
    
    # 3. Process the remaining documents in batches
    for i in tqdm(range(0, len(docs_to_process), BATCH_SIZE), total=total_batches, desc="Embedding locally"):
        batch = docs_to_process[i:i+BATCH_SIZE]
        texts = [doc["page_content"] for doc in batch]
        ids = [doc["metadata"]["url"] for doc in batch]
        metadatas = [doc["metadata"] for doc in batch]
        
        try:
            # Generate embeddings locally
            # normalize_embeddings=True is highly recommended for BGE and Cosine Similarity
            embeddings = model.encode(texts, normalize_embeddings=True)
            
            # ChromaDB does not allow empty lists or None values in metadata.
            # We must clean the metadatas before inserting.
            for meta in metadatas:
                keys_to_remove = []
                for k, v in meta.items():
                    # Check for empty lists or None
                    if v is None or (isinstance(v, list) and len(v) == 0):
                        keys_to_remove.append(k)
                for k in keys_to_remove:
                    del meta[k]
            
            # Save the generated embeddings into ChromaDB
            # tolist() converts the numpy array to the Python list of lists that ChromaDB expects
            collection.add(
                embeddings=embeddings.tolist(),
                documents=texts,
                metadatas=metadatas,
                ids=ids
            )
            
        except Exception as e:
            print(f"\nError processing batch {i // BATCH_SIZE}: {e}")

if __name__ == "__main__":
    main()
