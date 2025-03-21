
# 📚 BBC News Summarizer & Sentiment Analyzer

The **BBC News Summarizer & Sentiment Analyzer** is an intelligent web application designed to help users quickly digest large volumes of BBC news content. By entering a BBC topic URL or keyword, users can fetch multiple related articles, summarize them into concise snippets using powerful AI models, and analyze the overall sentiment of the coverage. This application leverages cutting-edge technologies such as 
**huggingface  facebook bart and sentiment analysis pipeline** for summarization and sentiment analysis.

---

## 📝 Introduction

This application is built to serve as a tool for researchers, media analysts, and curious readers who want to interpret news sentiment patterns and quickly understand the essence of lengthy articles. It provides a user-friendly interface using **Streamlit** and **FastAPI**, with seamless integration of AI capabilities.

---

## 🔎 Key Features

### Core Features:
- **Automatic Article Scraping:** Fetch BBC articles based on keywords or URLs.
- **AI-Powered Summarization:** Generate concise summaries using GROQ LLM models.
- **Sentiment Analysis:** Analyze the sentiment (positive, negative, neutral) of each article.
- **Topic Extraction:** Identify common keywords and themes across articles.
- **Article Comparisons:** 
  - Auto Comparison: Sentiment-based comparisons between articles.
  - Manual Comparison: Side-by-side comparison for in-depth analysis.
- **Hindi Sentiment Summary:** Sentiment summaries translated into Hindi for broader accessibility.

### Technical Features:
- Built with **FastAPI** and **Streamlit** for a robust backend and user-friendly frontend.
- Integrated with **LangChain** for advanced AI functionalities.
- Supports **GROQ API** for high-performance AI operations.

---

## 🎯 Project Goals

- **Sentiment Bias Detection:** Identify tone and bias in BBC's coverage of specific topics or companies.
- **Efficient Summarization:** Provide quick summaries of lengthy articles using AI.
- **Comprehensive Analysis:** Empower users to interpret sentiment patterns across multiple articles or timeframes.
- **User-Friendly Interface:** Deliver a smooth experience with **Streamlit** and **FastAPI**.

---

## 🚀 Tech Stack

| **Layer**            | **Technology**                                                    |
|----------------------|------------------------------------------------------------------|
| **Backend**          | FastAPI, BeautifulSoup, LangChain                               |
| **LLM**              | GROQ API (using **Mixtral** or **LLaMA2** models via LangChain) |
| **Frontend**         | Streamlit                                                       |
| **NLP Features**     |  huggingface pipeline + Sentiment Analysis + gtts                   |

---

## ⚡ Alternative Setup (Optional)

If you do not have access to **GROQ API**, you can modify the application to use **Hugging Face Transformers** models:

- **Summarization:** Using `facebook/bart-large-cnn`
- **Sentiment Analysis:** Using the `sentiment-analysis` pipeline from Hugging Face

### Example Usage:
```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
sentiment_analyzer = pipeline("sentiment-analysis")
```

---

## 📚 API Documentation

### 1. GET `/`
- **Description:** Returns a simple confirmation string.
- **Response:**
  ```json
  "string"
  ```

### 2. GET `/health`
- **Description:** Checks if the service is running.
- **Response:**
  ```json
  "Service is healthy"
  ```

### 3. POST `/analyze/`
- **Description:** Analyzes news articles for a specific keyword/company across multiple pages.
- **Request Body:**
  ```json
  {
    "query": "apple",
    "start_page": 1,
    "end_page": 3,
    "lines": 3
  }
  ```
- **Parameters:**
  | Parameter | Type | Description |
  | --- | --- | --- |
  | query | string | Keyword or company to search |
  | start_page | int | Starting page number |
  | end_page | int | Ending page number |
  | lines | int | Number of summary lines per article |

- **Example cURL:**
  ```bash
  curl -X 'POST' \
    'http://localhost:8000/analyze/' \
    -H 'accept: application/json' \
    -H 'Content-Type: application/json' \
    -d '{
      "query": "apple",
      "start_page": 1,
      "end_page": 3,
      "lines": 3
    }'
  ```

- **Example Response:**
  ```json
  {
    "success": true,
    "result": {
      "Company": "apple",
      "Articles": [
        {
          "Title": "Apple takes legal action in UK data privacy row",
          "Published Date": "4 March 2025",
          "Summary": "Apple has taken legal action to challenge...",
          "Sentiment": "Neutral",
          "Topics": [
            "Apple",
            "Advanced Data Protection",
            "Investigatory Powers Tribunal"
          ],
          "Link": "https://www.bbc.co.uk/news/articles/c8rkpv50x01o"
        }
      ]
    }
  }
  ```

---

## ⚙️ Project Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/your-repo-name.git
cd your-repo-name
```

### 2. Create a Virtual Environment
```bash
python -m venv venv
source venv/bin/activate   # On Windows: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure API Keys
Create a `.env` file in the project root:
```env
GROQ_API_KEY=your_groq_api_key_here
```

### 5. Run the FastAPI Server
```bash
uvicorn api:app --reload
```

### 6. Run the Streamlit App (Optional)
```bash
streamlit run app.py
```



## 📦 Dependencies

The project requires the following dependencies, listed in `requirements.txt`:
```text
altair==5.5.0
annotated-types==0.7.0
anyio==4.9.0
attrs==25.3.0
beautifulsoup4==4.13.3
blinker==1.9.0
cachetools==5.5.2
certifi==2025.1.31
charset-normalizer==3.4.1
click==8.1.8
dnspython==2.7.0
email_validator==2.2.0
exceptiongroup==1.2.2
fastapi==0.115.11
fastapi-cli==0.0.7
filelock==3.18.0
fsspec==2025.3.0
gitdb==4.0.12
GitPython==3.1.44
gTTS==2.5.4
h11==0.14.0
httpcore==1.0.7
httptools==0.6.4
httpx==0.28.1
huggingface-hub==0.29.3
idna==3.10
itsdangerous==2.2.0
Jinja2==3.1.6
jsonschema==4.23.0
jsonschema-specifications==2024.10.1
markdown-it-py==3.0.0
MarkupSafe==3.0.2
mdurl==0.1.2
mpmath==1.3.0
narwhals==1.31.0
networkx==3.4.2
numpy==2.2.4
orjson==3.10.15
packaging==24.2
pandas==2.2.3
pillow==11.1.0
protobuf==5.29.3
pyarrow==19.0.1
pydantic==2.10.6
pydantic-extra-types==2.10.3
pydantic-settings==2.8.1
pydantic_core==2.27.2
pydeck==0.9.1
Pygments==2.19.1
python-dateutil==2.9.0.post0
python-dotenv==1.0.1
python-multipart==0.0.20
pytz==2025.1
PyYAML==6.0.2
referencing==0.36.2
regex==2024.11.6
requests==2.32.3
rich==13.9.4
rich-toolkit==0.13.2
rpds-py==0.23.1
safetensors==0.5.3
shellingham==1.5.4
six==1.17.0
smmap==5.0.2
sniffio==1.3.1
soupsieve==2.6
starlette==0.46.1
streamlit==1.43.2
sympy==1.13.1
tenacity==9.0.0
tokenizers==0.21.1
toml==0.10.2
torch==2.6.0
tornado==6.4.2
tqdm==4.67.1
transformers==4.49.0
typer==0.15.2
typing_extensions==4.12.2
tzdata==2025.1
ujson==5.10.0
urllib3==2.3.0
uvicorn==0.34.0
uvloop==0.21.0
watchfiles==1.0.4
websockets==15.0.1

```

---

This README provides a comprehensive overview of the project, its features, and how to set it up. For further details or contributions, please refer to the GitHub repository.
