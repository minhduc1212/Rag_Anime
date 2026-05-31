import requests
from bs4 import BeautifulSoup
import re
import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor

URLS_FILE = 'anime_urls.json'
OUTPUT_FILE = 'all_anime_data.json'

file_lock = threading.Lock()

def get_data(url):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                html_content = response.text
                return parse_anime_data(html_content)
            elif response.status_code == 404:
                print(f"[Worker] 404 Not Found: {url}")
                return None
            else:
                print(f"[Worker] Error {response.status_code} fetching {url}. Retrying...")
                time.sleep(2)
        except Exception as e:
            print(f"[Worker] Exception {e} fetching {url}. Retrying...")
            time.sleep(2)
            
    return None

def parse_anime_data(html_content):
    soup = BeautifulSoup(html_content, 'html.parser')
    anime_data = {}

    # Title
    title_element = soup.select_one('h1.title-name')
    if title_element:
        anime_data['Title'] = title_element.text.strip()

    # Synopsis
    synopsis_element = soup.find('p', itemprop='description')
    if synopsis_element:
        synopsis = synopsis_element.text.strip()
        synopsis = re.sub(r'\n+', '\n', synopsis)
        anime_data['Synopsis'] = synopsis

    # Sidebar Information
    for div in soup.select('div.spaceit_pad'):
        dark_text_span = div.find('span', class_='dark_text')
        if dark_text_span:
            key = dark_text_span.text.strip().replace(':', '')
            dark_text_span.extract()
            
            a_tags = div.find_all('a')
            if a_tags and key in ['Producers', 'Studios', 'Genres', 'Themes', 'Demographic']:
                values = [a.text.strip() for a in a_tags if 'add some' not in a.text.lower()]
                values = list(dict.fromkeys(values))
                value = ', '.join(values)
            else:
                value = div.text.strip()
                value = re.sub(r'\s+', ' ', value)
                value = value.replace(' ,', ',')
                
            anime_data[key] = value

    # Additional cleanup for specific fields
    score_element = soup.find('span', itemprop='ratingValue')
    if score_element:
        anime_data['Score'] = score_element.text.strip()
    
    ranked_element = soup.select_one('span.numbers.ranked strong')
    if ranked_element:
        anime_data['Ranked'] = ranked_element.text.strip()
        
    popularity_element = soup.select_one('span.numbers.popularity strong')
    if popularity_element:
        anime_data['Popularity'] = popularity_element.text.strip()
        
    members_element = soup.select_one('span.numbers.members strong')
    if members_element:
        anime_data['Members'] = members_element.text.strip()

    return anime_data

def load_all_urls():
    if not os.path.exists(URLS_FILE):
        print(f"Error: {URLS_FILE} not found. Please run get_url.py first.")
        return []
    with open(URLS_FILE, 'r', encoding='utf-8') as f:
        return json.load(f)

def load_existing_data():
    if os.path.exists(OUTPUT_FILE):
        with open(OUTPUT_FILE, 'r', encoding='utf-8') as f:
            try:
                return json.load(f)
            except json.JSONDecodeError:
                return {}
    return {}

def save_data(data):
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        json.dump(data, f, indent=2, ensure_ascii=False)

def process_url(url, all_data, progress_counter):
    print(f"[Thread-{threading.get_ident()}] Fetching: {url}")
    data = get_data(url)
    
    if data is not None:
        data['url'] = url  # Keep the URL in the data for reference
        with file_lock:
            all_data[url] = data
            progress_counter[0] += 1
            
            # Save to file every 10 items to prevent huge I/O overhead on every single fetch
            if progress_counter[0] % 10 == 0:
                save_data(all_data)
                print(f"[*] Progress saved. Total records: {len(all_data)}")
                
    # Sleep slightly to avoid being IP banned
    time.sleep(1)

def main():
    urls = load_all_urls()
    if not urls:
        return
        
    all_data = load_existing_data()
    
    # Filter out URLs that have already been scraped successfully
    urls_to_process = [url for url in urls if url not in all_data]
    print(f"Total URLs: {len(urls)} | Already processed: {len(all_data)} | To process: {len(urls_to_process)}")
    
    progress_counter = [0] # Use a list to pass by reference to threads

    # Use 5 threads
    with ThreadPoolExecutor(max_workers=5) as executor:
        for url in urls_to_process:
            executor.submit(process_url, url, all_data, progress_counter)
            
    # Final save to catch any remaining data that didn't hit the modulo
    with file_lock:
        save_data(all_data)
    print(f"[*] Finished crawling! Total records in DB: {len(all_data)}")

if __name__ == "__main__":
    main()