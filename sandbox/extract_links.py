#!/usr/bin/env python3
import requests
from bs4 import BeautifulSoup

def extract_links(url):
    """Fetch a webpage and extract all links from it."""
    try:
        response = requests.get(url, timeout=10)
        response.raise_for_status()
    except Exception as e:
        print(f"Error fetching {url}: {e}")
        return

    soup = BeautifulSoup(response.text, 'html.parser')
    base_url = url.split('/')[3]  # Get website domain (thinksmart.life)
    all_links = []

    for a in soup.find_all('a', href=True):
        link = a['href']
        if link.startswith(('/', '//')):
            full_url = f"{base_url}{link}"
        else:
            full_url = f"{url.split('/')[0]}://{link}"
        print(full_url)

    print(f"\nFound {len(all_links)} links on {url}")

if __name__ == "__main__":
    url = "https://thinksmart.life"
    extract_links(url)