import re
import json
import time
import requests
from bs4 import BeautifulSoup
from gtts import gTTS
from urllib.parse import quote
from transformers import pipeline
from tqdm import tqdm
from pathlib import Path
import logging
# Setup logging
# Create root path
# Create root path
root_path = Path(__file__).parent
log_dir = root_path / 'log'
log_dir.mkdir(parents=True, exist_ok=True)  # Ensures the directory exists

# Set up logger
logger = logging.getLogger("utils log")
logger.setLevel(logging.DEBUG)

# Console handler
console_handler = logging.StreamHandler()
console_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(console_formatter)
logger.addHandler(console_handler)

# File handler
file_handler_path = log_dir / 'utils.log'
file_handler = logging.FileHandler(file_handler_path)
file_formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(file_formatter)
logger.addHandler(file_handler)

try:
    sentiment_analyzer = pipeline("sentiment-analysis")
    logger.info("Sentiment analysis pipeline loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load sentiment analysis pipeline: {e}")
    raise

try:
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn", device=0)
    logger.info("Summarization pipeline loaded successfully.")
except Exception as e:
    logger.error(f"Failed to load summarization pipeline: {e}")
    raise
def extract_topic_name(input_text: str) -> str:
    """
    Extracts a clean topic name by removing special characters.
    Used for naming files or directories.

    Args:
        input_text (str): Raw topic string or input text.

    Returns:
        str: Sanitized topic name.
    """
    try:
        if not isinstance(input_text, str) or not input_text.strip():
            logger.warning("Provided input_text is empty or not a string. Returning default 'topic'.")
            return "topic"

        logger.info(f"Extracting topic name from input: {input_text}")
        topic_name = re.sub(r'[^A-Za-z0-9_-]', '', input_text)

        if not topic_name:
            logger.warning(f"No valid characters found after sanitization. Returning default 'topic'.")
            return "topic"

        logger.info(f"Sanitized topic name: {topic_name}")
        return topic_name

    except Exception as e:
        logger.error(f"Error while extracting topic name: {e}")
        return "topic"

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
    if not topic_or_url or not isinstance(topic_or_url, str):
        logger.error("Invalid topic_or_url provided. It must be a non-empty string.")
        return []

    links = set()
    current_end = end_page

    logger.info(f"Starting article link extraction for: {topic_or_url}")

    while True:
        try:
            if topic_or_url.startswith("http"):
                logger.info(f"Fetching articles from URL: {topic_or_url}")
                response = requests.get(topic_or_url, timeout=10)
                response.raise_for_status()
                soup = BeautifulSoup(response.text, 'html.parser')
                page_links = [
                    f"https://www.bbc.com{a['href']}"
                    for a in soup.find_all('a', href=True)
                    if a['href'].startswith("/news/articles/")
                ]
                links.update(page_links)
                logger.info(f"Found {len(page_links)} articles from provided URL.")
            else:
                logger.info(f"Performing topic-based search for '{topic_or_url}' from page {start_page} to {current_end}.")
                for page in tqdm(range(start_page, current_end + 1), desc=f"Scraping pages {start_page}-{current_end}"):
                    search_url = f"https://www.bbc.co.uk/search?q={quote(topic_or_url)}&filter=news&page={page}"
                    response = requests.get(search_url, timeout=10)
                    response.raise_for_status()
                    soup = BeautifulSoup(response.text, 'html.parser')
                    page_links = [
                        a['href']
                        for a in soup.find_all('a', href=True)
                        if '/news/articles/' in a['href']
                    ]
                    links.update(page_links)
                    logger.info(f"Page {page}: Found {len(page_links)} articles.")

            if len(links) >= min_articles or topic_or_url.startswith("http"):
                logger.info(f"Article collection complete. Found {len(links)} articles.")
                break
            else:
                current_end += 2
                logger.warning(f"Only {len(links)} articles found. Expanding search up to page {current_end}.")
                time.sleep(1)

        except requests.exceptions.RequestException as e:
            logger.error(f"HTTP request failed: {e}")
            break
        except Exception as e:
            logger.error(f"Unexpected error during scraping: {e}")
            break

    return list(links)[:min_articles]

def get_article_content(article_url: str) -> tuple:
    """
    Fetches article headline, publish date, and full text content.
    
    Args:
        article_url (str): URL of the article to scrape.
        
    Returns:
        tuple: (headline, publication_date, article_text)
    """
    try:
        logger.info(f"Fetching content from: {article_url}")
        response = requests.get(article_url, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.text, 'html.parser')
        
        # Headline
        headline_tag = soup.find("h1")
        headline = headline_tag.text.strip() if headline_tag else "No headline found"
        
        # Publication date
        pub_date_tag = soup.find("time")
        pub_date = pub_date_tag.text.strip() if pub_date_tag else "No publish date found"
        
        # Article body text
        paragraphs = soup.find_all("p")
        if not paragraphs:
            logger.warning(f"No <p> tags found for article: {article_url}")
        article_text = " ".join([p.get_text(strip=True) for p in paragraphs if p.get_text(strip=True)])
        
        if not article_text:
            logger.warning(f"No article text extracted from: {article_url}")

        logger.info(f"Scraped article: '{headline}' (Published: {pub_date})")
        return headline, pub_date, article_text

    except requests.exceptions.RequestException as e:
        logger.error(f"HTTP request failed for {article_url}: {e}")
        return "Error fetching article", "N/A", ""
    except Exception as e:
        logger.error(f"Unexpected error while scraping article from {article_url}: {e}")
        return "Error fetching article", "N/A", ""
    
    
def summarize_text(text: str, desired_lines: int = 3) -> str:
    """
    Summarizes long text into a concise summary using a transformer-based model.
    
    Args:
        text (str): Text to summarize.
        desired_lines (int): Rough target number of lines for summary.
        
    Returns:
        str: Summarized text or fallback message.
    """
    try:
        if not text or len(text.strip()) == 0:
            logger.warning("Empty text provided for summarization.")
            return "No content to summarize."

        # Limit text length for the model (1024 words max)
        word_limit = 1024
        words = text.split()
        if len(words) > word_limit:
            logger.warning(f"Input text exceeds {word_limit} words. Truncating for summarization.")
            text = " ".join(words[:word_limit])

        # Calculate dynamic summary length
        max_length = min(150, desired_lines * 25)
        min_length = max(10, desired_lines * 10)

        logger.info(f"Generating summary (min_length={min_length}, max_length={max_length})")
        summary_output = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)
        
        summary_text = summary_output[0]['summary_text']
        logger.info("Summary generated successfully.")
        return summary_text

    except Exception as e:
        logger.error(f"Error during summarization: {e}")
        return "Summary generation failed due to an unexpected error."

import logging
import re

logger = logging.getLogger("topic_extractor")

def extract_topics(summary: str) -> list:
    """
    Extracts proper noun phrases or potential topics from the generated summary.
    
    Args:
        summary (str): Text summary from which to extract topics.
        
    Returns:
        list: List of up to 5 extracted topic keywords.
    """
    try:
        if not summary or len(summary.strip()) == 0:
            logger.warning("Empty summary provided for topic extraction.")
            return ["No topics found"]

        logger.info("Extracting topics from summary...")
        keywords = re.findall(r'\b[A-Z][a-z]+(?: [A-Z][a-z]+)*\b', summary)

        # Remove duplicates while preserving order
        seen = set()
        unique_keywords = [kw for kw in keywords if not (kw in seen or seen.add(kw))]

        extracted_topics = unique_keywords[:5]
        if not extracted_topics:
            logger.warning("No topics identified in the summary.")
            return ["No topics found"]

        logger.info(f"Extracted topics: {extracted_topics}")
        return extracted_topics

    except Exception as e:
        logger.error(f"Error while extracting topics: {e}")
        return ["Topic extraction failed"]


def create_hindi_tts(text: str, filename: str) -> None:
    """
    Generates a Hindi audio file from given text using Google Text-to-Speech (gTTS).
    
    Args:
        text (str): Text to convert to speech.
        filename (str): Output audio file name (e.g., 'output.mp3').
    """
    try:
        if not text.strip():
            logger.warning("Empty text provided. Skipping TTS generation.")
            return

        if not filename.endswith('.mp3'):
            logger.warning("Output filename does not end with '.mp3'. Appending extension.")
            filename += '.mp3'

        logger.info("Generating Hindi TTS audio...")
        tts = gTTS(text=text, lang='hi')
        tts.save(filename)
        logger.info(f"Hindi audio saved as: {filename}")

    except Exception as e:
        logger.error(f"Failed to generate TTS audio: {e}")

from pathlib import Path

# Create a results folder if it doesn't exist
RESULTS_DIR = Path("results")
RESULTS_DIR.mkdir(exist_ok=True)

def save_analysis_json(result: dict, company_name: str) -> None:
    """
    Saves the analysis result as a JSON file inside the results folder.
    
    Args:
        result (dict): Analysis result to be saved.
        company_name (str): The name of the company/topic for naming the file.
    """
    try:
        sanitized_name = re.sub(r'[^A-Za-z0-9_-]', '', company_name.lower())
        file_name = RESULTS_DIR / f"{sanitized_name}_summary.json"

        logger.info(f"Saving analysis results for '{company_name}'...")
        with open(file_name, "w", encoding='utf-8') as file:
            json.dump(result, file, indent=4, ensure_ascii=False)
        logger.info(f" Results saved at: {file_name.resolve()}")

    except Exception as e:
        logger.error(f" Failed to save analysis JSON for {company_name}: {e}")

def create_hindi_tts(text: str, filename: str) -> None:
    """
    Generates a Hindi audio file from given text and saves it inside the results folder.

    Args:
        text (str): Text to convert to speech.
        filename (str): Output audio file name (e.g., 'summary.mp3').
    """
    try:
        audio_path = RESULTS_DIR / filename
        logger.info(f"Generating Hindi TTS audio for: {audio_path}")
        tts = gTTS(text=text, lang='hi')
        tts.save(audio_path)
        logger.info(f" Hindi audio saved at {audio_path.resolve()}")
    except Exception as e:
        logger.error(f" Failed to create Hindi TTS audio: {e}")

#  Function to compare two articles by their indices
def compare_articles(articles: list, index1: int, index2: int) -> dict:
    """
    Compare two articles by their indices and return a summary dictionary.

    Args:
        articles (list of dict): List of articles with 'Title' and 'Sentiment' keys.
        index1 (int): Index of the first article.
        index2 (int): Index of the second article.

    Returns:
        dict: A summary comparing the two articles.
    """
    try:
        if not articles or not isinstance(articles, list):
            raise ValueError("Articles should be a non-empty list of dictionaries.")
        
        a1 = articles[index1]
        a2 = articles[index2]
        
        comparison = {
            "Article 1": a1.get("Title", "No Title"),
            "Article 2": a2.get("Title", "No Title"),
            "Article 1 Sentiment": a1.get("Sentiment", "Unknown"),
            "Article 2 Sentiment": a2.get("Sentiment", "Unknown"),
            "Comparison Summary": (
                f"'{a1.get('Title', 'Article 1')}' has a sentiment of {a1.get('Sentiment', 'Unknown')}, "
                f"while '{a2.get('Title', 'Article 2')}' has a sentiment of {a2.get('Sentiment', 'Unknown')}."
            )
        }
        logger.info(f"Comparison completed between article {index1} and article {index2}.")
        return comparison
    
    except IndexError:
        logger.error(f"Index out of range. Please provide indices within 0 and {len(articles)-1}.")
        return {"error": "Index out of range."}
    except Exception as e:
        logger.error(f"Error comparing articles: {e}")
        return {"error": str(e)}


from collections import Counter

def analyze_articles(links: list, company_name: str, desired_lines: int = 3) -> dict:
    """
    Processes each article by summarizing content, analyzing sentiment, extracting topics,
    and generating comparative insights and sentiment conclusions.

    Args:
        links (list): List of article URLs.
        company_name (str): The company or topic name being analyzed.
        desired_lines (int): Desired number of lines in each article's summary.

    Returns:
        dict: The complete analysis result.
    """
    start_time = time.time()
    articles = []
    sentiments = []

    for link in tqdm(links, desc="Analyzing articles"):
        try:
            headline, pub_date, text = get_article_content(link)
            combined_text = f"{headline}. {text}"
            summary = summarize_text(combined_text, desired_lines)
            sentiment = sentiment_analyzer(summary)[0]
            topics = extract_topics(summary)

            article_data = {
                "Title": headline,
                "Published Date": pub_date,
                "Summary": summary,
                "Sentiment": sentiment['label'],
                "Topics": topics,
                "Link": link
            }

            articles.append(article_data)
            sentiments.append(sentiment['label'])
            # Optional: add random sleep if needed
            # time.sleep(random.uniform(0.5, 1.5))

        except Exception as e:
            logging.warning(f"⚠️ Error processing {link}: {e}")

    if not articles:
        logging.error(f"No articles could be processed for {company_name}.")
        return {}

    sentiment_counts = dict(Counter(sentiments))

    comparisons = [
        {
            "Comparison": f"{a1['Title']} vs. {a2['Title']}",
            "Impact": f"Article 1 sentiment: {a1['Sentiment']}, Article 2 sentiment: {a2['Sentiment']}"
        }
        for i, a1 in enumerate(articles)
        for j, a2 in enumerate(articles)
        if i < j
    ][:10]

    topic_sets = [set(a['Topics']) for a in articles if a['Topics']]
    common_topics = list(set.intersection(*topic_sets)) if len(topic_sets) > 1 else []
    unique_topics = [
        list(topics - set(common_topics)) for topics in topic_sets
    ] if topic_sets else []

    final_sentiment = max(sentiment_counts, key=sentiment_counts.get).lower()
    hindi_summary_text = f"{company_name} की खबरों का समग्र झुकाव {final_sentiment} है।"

    result = {
        "Company": company_name,
        "Articles": articles,
        "Comparative Sentiment Insights": {
            "Sentiment Distribution": sentiment_counts,
            "Comparisons": comparisons,
            "Topic Analysis": {
                "Common Topics": common_topics,
                "Unique Topics per Article": unique_topics
            }
        },
        "Overall Sentiment Conclusion": f"{company_name}'s news coverage is {final_sentiment}.",
        "Hindi Sentiment Summary": hindi_summary_text,
        "Hindi_TTS_Audio_File": f"{company_name.lower()}_sentiment_hindi.mp3"
    }

    # Save results and create audio with exception handling
    try:
        save_analysis_json(result, company_name)
        create_hindi_tts(hindi_summary_text, f"{company_name.lower()}_sentiment_hindi.mp3")
    except Exception as e:
        logging.error(f"Error while saving analysis or generating audio: {e}")

    elapsed_time = time.time() - start_time
    logging.info(f"✅ Analysis completed for {company_name} in {elapsed_time:.2f} seconds.")
    logging.info(json.dumps(result, indent=4, ensure_ascii=False))

    return result