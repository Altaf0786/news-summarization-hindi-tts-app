import re
import json
import time
import requests
from bs4 import BeautifulSoup
from gtts import gTTS
from urllib.parse import quote
from transformers import pipeline
from tqdm import tqdm

# Initialize NLP pipelines once at load time (for efficiency)
sentiment_analyzer = pipeline("sentiment-analysis")
summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)

def extract_topic_name(input_text: str) -> str:
    """
    Extracts a clean topic name by removing special characters.
    Used for naming files or directories.
    
    Args:
        input_text (str): Raw topic string or input text.
        
    Returns:
        str: Sanitized topic name.
    """
    topic_name = re.sub(r'[^A-Za-z0-9_-]', '', input_text)
    return topic_name if topic_name else "topic"

def get_article_links(topic_or_url: str, start_page: int = 1, end_page: int = 3, min_articles: int = 10) -> list:
    """
    Collects article links from either a topic search or a specific URL.
    Dynamically increases search pages if required.
    
    Args:
        topic_or_url (str): Search topic or a direct BBC URL.
        start_page (int): Start page for pagination.
        end_page (int): Initial end page for pagination.
        min_articles (int): Minimum number of articles to retrieve.
        
    Returns:
        list: List of article URLs.
    """
    links = set()
    current_end = end_page
    
    while True:
        if topic_or_url.startswith("http"):
            # If a URL is provided, scrape articles directly
            response = requests.get(topic_or_url)
            soup = BeautifulSoup(response.text, 'html.parser')
            page_links = [
                f"https://www.bbc.com{a['href']}"
                for a in soup.find_all('a', href=True)
                if a['href'].startswith("/news/articles/")
            ]
            links.update(page_links)
        else:
            # Perform topic-based search across multiple pages
            for page in tqdm(range(start_page, current_end + 1), desc=f"Scraping pages {start_page}-{current_end}"):
                search_url = f"https://www.bbc.co.uk/search?q={quote(topic_or_url)}&filter=news&page={page}"
                response = requests.get(search_url)
                soup = BeautifulSoup(response.text, 'html.parser')
                page_links = [
                    a['href']
                    for a in soup.find_all('a', href=True)
                    if '/news/articles/' in a['href']
                ]
                links.update(page_links)
        
        # Check if sufficient articles have been collected
        if len(links) >= min_articles or topic_or_url.startswith("http"):
            break
        else:
            # Expand search if not enough articles found
            current_end += 2
            print(f"⚠️ Only found {len(links)} articles, expanding search to page {current_end}...")
            time.sleep(1)

    return list(links)[:min_articles]

def get_article_content(article_url: str) -> tuple:
    """
    Fetches article headline, publish date, and full text content.
    
    Args:
        article_url (str): URL of the article to scrape.
        
    Returns:
        tuple: (headline, publication_date, article_text)
    """
    response = requests.get(article_url)
    soup = BeautifulSoup(response.text, 'html.parser')
    headline = soup.find("h1").text.strip() if soup.find("h1") else "No headline"
    pub_date = soup.find("time").text.strip() if soup.find("time") else "No publish date"
    article_text = " ".join([p.text.strip() for p in soup.find_all("p")])
    return headline, pub_date, article_text

def summarize_text(text: str, desired_lines: int = 3) -> str:
    """
    Summarizes long text into a concise summary using a transformer-based model.
    
    Args:
        text (str): Text to summarize.
        desired_lines (int): Rough target number of lines for summary.
        
    Returns:
        str: Summarized text.
    """
    # Limit text length for the model (1024 words max)
    word_limit = 1024
    text = text[:word_limit]

    # Calculate dynamic lengths
    max_length = min(150, desired_lines * 25)
    min_length = max(10, desired_lines * 10)

    # Generate summary
    summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
    return summary[0]['summary_text']

def extract_topics(summary: str) -> list:
    """
    Extracts proper noun phrases or potential topics from the generated summary.
    
    Args:
        summary (str): Text summary from which to extract topics.
        
    Returns:
        list: List of up to 5 extracted topic keywords.
    """
    keywords = re.findall(r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b', summary)
    return list(set(keywords))[:5]

def create_hindi_tts(text: str, filename: str) -> None:
    """
    Generates a Hindi audio file from given text using Google Text-to-Speech.
    
    Args:
        text (str): Text to convert to speech.
        filename (str): Output audio file name (e.g., 'output.mp3').
    """
    tts = gTTS(text=text, lang='hi')
    tts.save(filename)
    print(f"Hindi audio saved as {filename}")

def save_analysis_json(result: dict, company_name: str) -> None:
    """
    Saves the analysis result as a JSON file for easy retrieval and reuse.
    
    Args:
        result (dict): Dictionary containing analysis results.
        company_name (str): Used for naming the output file.
    """
    file_name = f"{company_name.lower()}_summary.json"
    with open(file_name, "w", encoding='utf-8') as file:
        json.dump(result, file, indent=4, ensure_ascii=False)
    print(f"✅ Results saved in {file_name}")
