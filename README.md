# 🧬 Smart Biology Assistant Chatbot

A Flask-based conversational chatbot that answers biology questions using a hybrid NLP search pipeline backed by a curated Q&A dataset, with automatic fallback to Google Gemini AI for questions outside the dataset.

---

## Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Architecture](#architecture)
- [Project Structure](#project-structure)
- [Getting Started](#getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Configuration](#configuration)
  - [Running Locally](#running-locally)
- [Deployment](#deployment)
- [How It Works](#how-it-works)
- [Dataset](#dataset)
- [Technologies Used](#technologies-used)

---

## Overview

Smart Biology Assistant is a web-based chatbot that helps students and learners get instant answers to biology questions. It combines classical information-retrieval techniques (BM25 + TF-IDF) with semantic similarity scoring to find the best matching answer from its knowledge base. When no strong match is found, it falls back to Google's Gemini 1.5 Flash model to generate a concise answer.

---

## Features

- **Hybrid search** – combines BM25 keyword ranking and TF-IDF cosine similarity for high-recall retrieval.
- **Semantic re-ranking** – uses spaCy word vectors to score candidate answers by semantic closeness to the query.
- **Conversation context** – tracks noun-phrase keywords across turns to improve multi-turn coherence.
- **Gemini AI fallback** – seamlessly calls the Gemini 1.5 Flash API when the local dataset has no strong match.
- **Typing animation** – bot responses appear with a character-by-character typing effect.
- **Chat history sidebar** – previous user messages are listed in the sidebar for easy recall.
- **Suggested questions** – quick-tap prompts help users explore common biology topics.
- **Responsive UI** – layout adapts to mobile screens.

---

## Architecture

```
User (browser)
      │  HTTP POST /ask  { message }
      ▼
Flask app (app.py)
      │
      ├─ preprocess(query)          # spaCy lemmatization + stopword removal
      ├─ BM25 top-5 candidates      # keyword-level retrieval
      ├─ TF-IDF top-5 candidates    # n-gram cosine similarity retrieval
      ├─ Union of candidates
      ├─ Semantic re-rank           # spaCy doc.similarity + keyword history boost
      │
      ├─ score ≥ 0.4 → return best answer from bio.csv
      └─ score < 0.4 → Gemini 1.5 Flash API → return generated answer
```

---

## Project Structure

```
Chatbot/
├── app.py                  # Flask application and NLP pipeline
├── bio.csv                 # Curated biology Q&A dataset (~547 pairs)
├── intent_classifier.pkl   # Serialised intent classifier (future use)
├── templates/
│   └── index.html          # Single-page chat UI
├── requirements.txt        # Python dependencies
├── Procfile                # Gunicorn entry point for Heroku / Railway
├── .env                    # Environment variables (not committed to VCS)
└── gitignore               # Git ignore rules
```

---

## Getting Started

### Prerequisites

- Python 3.9+
- A [Google Gemini API key](https://aistudio.google.com/app/apikey)

### Installation

```bash
# 1. Clone the repository
git clone https://github.com/amancodes8/Chatbot.git
cd Chatbot

# 2. Create and activate a virtual environment
python -m venv venv
source venv/bin/activate   # Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Download the spaCy language model
python -m spacy download en_core_web_lg
```

### Configuration

Create a `.env` file in the project root:

```
GEMINI_API_KEY=your_gemini_api_key_here
```

### Running Locally

```bash
python app.py
```

Open your browser at `http://127.0.0.1:5000`.

---

## Deployment

The project includes a `Procfile` for deployment on platforms such as Heroku or Railway:

```
web: gunicorn app:app
```

Set the `GEMINI_API_KEY` environment variable in your platform's dashboard before deploying.

---

## How It Works

1. **Preprocessing** – The user query and every question in `bio.csv` are lowercased, lemmatized, and stripped of stopwords and punctuation using spaCy. A separate keyword-preserving variant retains noun-phrase chunks.

2. **Hybrid retrieval** – BM25 scores keyword overlap between the query and each dataset question, while TF-IDF with 1–3-gram features captures phrase-level similarity. The top-5 candidates from each method are merged.

3. **Semantic re-ranking** – For each candidate answer, a combined score is computed:
   - **Keyword history boost (×0.7)** – counts how many noun phrases from the conversation history appear in the answer.
   - **Semantic similarity (×0.3)** – spaCy vector similarity between the answer and the query.

4. **Threshold check** – If the best score is ≥ 0.4, that answer is returned. Otherwise the query is forwarded to the Gemini 1.5 Flash API which replies in 1–2 concise sentences.

5. **Conversation context** – A `ConversationContext` object accumulates noun-phrase keywords from every user turn, letting the re-ranker progressively weight answers that relate to the ongoing topic.

---

## Dataset

`bio.csv` contains approximately 547 deduplicated biology Q&A pairs covering topics such as:

- Nervous and endocrine systems
- Photosynthesis and cellular respiration
- Genetics, chromosomes, and heredity
- Evolution and natural selection
- Circulatory, digestive, and reproductive systems
- Cell biology and ecology

Each row has three columns: `Intent`, `Question`, and `Answer`.

---

## Technologies Used

| Layer | Library / Service |
|---|---|
| Web framework | Flask 3.1 |
| NLP | spaCy 3.8 (`en_core_web_lg`) |
| Keyword retrieval | rank-bm25 |
| Vector retrieval | scikit-learn TF-IDF |
| AI fallback | Google Gemini 1.5 Flash (`google-generativeai`) |
| Production server | Gunicorn |
| Environment config | python-dotenv |
