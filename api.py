from fastapi import FastAPI
from pydantic import BaseModel
from typing import List
from utils import get_article_links, analyze_articles, extract_topic_name, compare_articles
import json
from pathlib import Path

# Initialize FastAPI app
app = FastAPI(
    title="News Summarization and Comparison API",
    description="Analyze articles based on query and compare article summaries.",
    version="1.0.0"
)

# Request model for /analyze endpoint
class AnalyzeRequest(BaseModel):
    query: str
    start_page: int
    end_page: int
    lines: int

# Request model for /compare endpoint
class CompareRequest(BaseModel):
    articles: List[dict]
    index1: int
    index2: int

# Root endpoint - just to confirm API is running
@app.get("/")
def root():
    return {"message": "API is running! Go to /docs for Swagger UI."}

# Health check endpoint (optional)
@app.get("/health")
def health_check():
    return {"status": "ok"}

# Endpoint for analyzing articles and returning summaries
@app.post("/analyze/")
def analyze(request_data: AnalyzeRequest):
    company_name = extract_topic_name(request_data.query)
    article_links = get_article_links(
        request_data.query, 
        request_data.start_page, 
        request_data.end_page, 
        min_articles=10
    )

    if article_links:
        analyze_articles(article_links, company_name, request_data.lines)
        json_file = Path(f"results/{company_name.lower()}_summary.json")
        if json_file.exists():
            with json_file.open("r", encoding="utf-8") as f:
                result = json.load(f)
            return {
                "success": True, 
                "result": result, 
                "company_name": company_name
            }
        else:
            return {"success": False, "message": "Summary file not found."}
    else:
        return {"success": False, "message": "No articles found for this query."}

# Endpoint for comparing two articles from a provided list
@app.post("/compare/")
def compare(compare_data: CompareRequest):
    comparison_result = compare_articles(
        compare_data.articles, 
        compare_data.index1, 
        compare_data.index2
    )
    return comparison_result
