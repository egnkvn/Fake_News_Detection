import requests
from readability import Document
from bs4 import BeautifulSoup
from newspaper import Article
from duckduckgo_search import DDGS
from datetime import datetime, date as _date

def get_publish_date(url):
    try:
        art = Article(url)
        art.download(); art.parse()
        if art.publish_date:
            return art.publish_date.date()
    except Exception:
        pass
    return None
    
def parse_website(url, timeout=10):
    text = ""
    try:
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        doc = Document(response.text)
        content_html = doc.summary()
        soup = BeautifulSoup(content_html, "html.parser")
        text = soup.get_text(separator=" ").strip()
    except Exception as e:
        pass
        # print(f"[readability] Error fetching {url}: {e}")
    return text


def search_online(title, cutoff_date: _date, num_results=5):
    results = DDGS().text(title, region='tw-tzh', max_results=25)
    results = [result for result in results if not result['href'].lower().endswith('.pdf')]
    urls = [r["href"] for r in results]

    articles = []
    for url in urls:
        pub = get_publish_date(url)
        if pub and pub > cutoff_date:
            continue
        text = parse_website(url)
        if text:
            articles.append(text)
        if len(articles) >= num_results:
            break

    return articles

if __name__ == "__main__":
    # Example usage
    title = "美國秘密推動「量子異質晶片」計劃，挑戰亞洲半導體霸主地位"
    date = "2025-05-23"

    cutoff = datetime.fromisoformat(date).date()
    articles = search_online(title, cutoff, num_results=5)
    for i, article in enumerate(articles):
        print(f"Article {i+1}:\n{article}\n")