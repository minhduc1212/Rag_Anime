import requests
from bs4 import BeautifulSoup
import json
import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor

URLS_FILE = 'anime_urls.json'
LOG_FILE = 'crawler_progress.json'
BASE_URL = 'https://myanimelist.net/anime.php?letter={}&show={}'

file_lock = threading.Lock()

def load_progress():
    if os.path.exists(LOG_FILE):
        with open(LOG_FILE, 'r', encoding='utf-8') as f:
            return json.load(f)
    return {}

def save_progress(progress):
    with file_lock:
        with open(LOG_FILE, 'w', encoding='utf-8') as f:
            json.dump(progress, f, indent=2)

def load_urls():
    if os.path.exists(URLS_FILE):
        with open(URLS_FILE, 'r', encoding='utf-8') as f:
            return set(json.load(f))
    return set()

def save_urls(urls_set):
    with file_lock:
        with open(URLS_FILE, 'w', encoding='utf-8') as f:
            json.dump(list(urls_set), f, indent=2)

def get_urls_from_page(letter, show):
    url = BASE_URL.format(letter, show)
    print(f"[Thread-Letter-{letter}] Fetching {url}...")
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
    }
    
    max_retries = 3
    for attempt in range(max_retries):
        try:
            response = requests.get(url, headers=headers, timeout=10)
            if response.status_code == 200:
                soup = BeautifulSoup(response.text, 'html.parser')
                container = soup.select_one('div.js-categories-seasonal.js-block-list.list')
                if not container:
                    return []
                
                links = container.find_all('a')
                urls = []
                for a in links:
                    href = a.get('href')
                    if href and href.startswith('https://myanimelist.net/anime/') and not href.endswith('video'):
                        urls.append(href)
                return list(set(urls))
            elif response.status_code == 404:
                return []
            else:
                print(f"[Thread-Letter-{letter}] Error {response.status_code}. Retrying...")
                time.sleep(2)
        except Exception as e:
            print(f"[Thread-Letter-{letter}] Exception {e}. Retrying...")
            time.sleep(2)
            
    return []

def process_letter(letter, progress, urls):
    # This runs in a separate thread for each letter
    with file_lock:
        if letter not in progress:
            progress[letter] = {'show': 0, 'done': False}
        is_done = progress[letter]['done']
        show = progress[letter]['show']
        
    if is_done:
        print(f"[Thread-Letter-{letter}] Letter is already done. Skipping.")
        return

    while True:
        new_urls = get_urls_from_page(letter, show)
        if not new_urls:
            print(f"[Thread-Letter-{letter}] No more URLs found at show={show}. Marking as done.")
            with file_lock:
                progress[letter]['done'] = True
            save_progress(progress)
            break
        
        # Save new urls
        with file_lock:
            initial_count = len(urls)
            urls.update(new_urls)
            total_count = len(urls)
            
            # Update progress
            show += 50
            progress[letter]['show'] = show
            
        # Write to files holding the lock inside the save functions
        save_urls(urls)
        save_progress(progress)
        
        print(f"[Thread-Letter-{letter}] Added {len(new_urls)} URLs (Total unique: {total_count}). Next show: {show}")
        
        # Sleep to be polite to the server
        time.sleep(1.5)

def main():
    letters = ['.'] + [chr(i) for i in range(ord('A'), ord('Z')+1)]
    progress = load_progress()
    urls = load_urls()
    
    # We use ThreadPoolExecutor to run 5 threads concurrently.
    # Each thread will process an entire letter start to finish.
    with ThreadPoolExecutor(max_workers=5) as executor:
        for letter in letters:
            executor.submit(process_letter, letter, progress, urls)
            
    print("All crawling tasks are finished!")

if __name__ == "__main__":
    main()