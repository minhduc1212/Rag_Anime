import json

def clean_number(val):
    if not val or val == "N/A" or val == "Unknown" or val == "-":
        return None
    if isinstance(val, str):
        val = val.replace("#", "").replace(",", "").strip()
        try:
            if "." in val:
                return float(val)
            return int(val)
        except ValueError:
            return None
    return val

def clean_list(val):
    if not val or val == "None found, add some" or val == "Unknown" or val == "-":
        return []
    return [item.strip() for item in val.split(",") if item.strip()]

def process_data(input_file, output_file):
    with open(input_file, 'r', encoding='utf-8') as f:
        anime_data = json.load(f)
        
    documents = []
    
    for url, data in anime_data.items():
        title = data.get("Title", "").strip()
        synopsis = data.get("Synopsis", "").strip()
        
        # Skip if title or synopsis is missing (or very short)
        if not title or not synopsis or "No synopsis information has been added" in synopsis:
            continue
            
        # Clean fields
        score = clean_number(data.get("Score"))
        ranked = clean_number(data.get("Ranked"))
        popularity = clean_number(data.get("Popularity"))
        members = clean_number(data.get("Members"))
        favorites = clean_number(data.get("Favorites"))
        episodes = clean_number(data.get("Episodes"))
        
        genres = clean_list(data.get("Genres"))
        studios = clean_list(data.get("Studios"))
        
        # Create page content for embedding
        # We want the content to be rich for semantic search
        alt_titles = []
        for alt in ["Synonyms", "Japanese", "English", "Spanish", "French", "German"]:
            if data.get(alt):
                alt_titles.append(data[alt])
        
        alt_titles_str = ", ".join(alt_titles) if alt_titles else "N/A"
        
        content_parts = [
            f"Title: {title}",
            f"Alternative Titles: {alt_titles_str}" if alt_titles else "",
            f"Genres: {', '.join(genres) if genres else 'N/A'}",
            f"Type: {data.get('Type', 'N/A')}, Episodes: {data.get('Episodes', 'N/A')}",
            f"Status: {data.get('Status', 'N/A')}, Aired: {data.get('Aired', 'N/A')}",
            f"Studios: {', '.join(studios) if studios else 'N/A'}",
            f"Score: {data.get('Score', 'N/A')} (Ranked: {data.get('Ranked', 'N/A')}, Popularity: {data.get('Popularity', 'N/A')})",
            f"Synopsis: {synopsis}"
        ]
        
        page_content = "\n".join([p for p in content_parts if p and not p.endswith("N/A") and "N/A (Ranked: N/A" not in p])
        
        # Metadata for filtering
        metadata = {
            "title": title,
            "url": url,
            "score": score if score is not None else 0.0,
            "ranked": ranked if ranked is not None else 999999,
            "popularity": popularity if popularity is not None else 999999,
            "members": members if members is not None else 0,
            "favorites": favorites if favorites is not None else 0,
            "episodes": episodes if episodes is not None else 0,
            "type": data.get("Type", "Unknown"),
            "status": data.get("Status", "Unknown"),
            "genres": genres,
            "studios": studios
        }
        
        documents.append({
            "page_content": page_content,
            "metadata": metadata
        })
        
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(documents, f, indent=2, ensure_ascii=False)
        
    print(f"Processed {len(anime_data)} entries.")
    print(f"Created {len(documents)} structured documents.")

if __name__ == "__main__":
    process_data('all_anime_data.json', 'cleaned_anime_docs.json')