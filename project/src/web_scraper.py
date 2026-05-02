import requests
from bs4 import BeautifulSoup
import pandas as pd
import os
import time
import warnings
from bs4 import XMLParsedAsHTMLWarning

# Suppress the HTML parser warning for RSS feeds
warnings.filterwarnings("ignore", category=XMLParsedAsHTMLWarning)

def scrape_ev_news():
    """
    Scrapes the InsideEVs news feed for recent Electric Vehicle articles.
    Extracts titles, descriptions, and publication dates to build a custom dataset.
    """
    print("--- Starting Task 1: Web Scraping ---")
    
    # We use an RSS feed URL as it's typically more stable for scraping without anti-bot blocks
    url = "https://insideevs.com/rss/news/all/"
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status() # Check if request was successful
        
        # Use html.parser which is built-in
        soup = BeautifulSoup(response.content, features="html.parser")
        
        items = soup.find_all("item")
        
        scraped_data = []
        for item in items:
            title = item.title.text if item.title else "No Title"
            link = item.link.text if item.link else "No Link"
            pub_date = item.pubDate.text if item.pubDate else "No Date"
            description = item.description.text if item.description else "No Description"
            
            # Clean up the CDATA if present or HTML tags in description
            desc_soup = BeautifulSoup(description, "html.parser")
            clean_desc = desc_soup.get_text(separator=" ", strip=True)
            
            scraped_data.append({
                "title": title,
                "description": clean_desc,
                "pub_date": pub_date,
                "link": link
            })
            
        return scraped_data
            
    except Exception as e:
        print(f"Error occurred while scraping: {e}")
        return []

def run_scraper():
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    data_dir = os.path.join(base_dir, "data")
    os.makedirs(data_dir, exist_ok=True)
    
    data = scrape_ev_news()
    
    if data:
        df = pd.DataFrame(data)
        out_path = os.path.join(data_dir, "ev_news_scraped.csv")
        df.to_csv(out_path, index=False)
        print(f"Successfully scraped {len(df)} articles.")
        print(f"Saved custom dataset to: {out_path}")
    else:
        print("Failed to gather scraped data.")

if __name__ == "__main__":
    run_scraper()
