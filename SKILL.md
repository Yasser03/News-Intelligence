---
name: News Intelligence System
description: Classify news articles and identify incident-related content using ML/LLMs.
---

# News Intelligence System

A scalable news intelligence collection and analysis system that classifies articles and identifies security/incident-related content. It uses a hybrid approach with Machine Learning (Logistic Regression) for fast classification and LLMs for contextual analysis.

## Prerequisites

- **Python 3.10+**
- **Dependencies**: Install via `pip install -r requirements.txt`
- **Models**: Ensure model weights exist in `models/`
  - `Logistic Regression_62k.pkl`
  - `label_encoder_transformer.pkl`
  - `word_vectorizer.pkl`

---

## Core Capabilities

### 1. News Binary Classifier

The `NewsBinaryClassifier` provides optimized, vectorized classification for large batches of news articles. It distinguishes between "incident" and "non-incident" content.

**Supported Modes:**
- **Full Text**: Uses `title` + `description` (Best accuracy)
- **Headline-only**: Uses `title` only (Fastest, 2-3x speedup)
- **With Probability**: Returns prediction + confidence score (0-1)

#### Required DataFrame Columns:
- `title`: Article headline/title
- `description`: Article summary (optional for headline-only mode)
- `article`: Full article text (optional for headline-only mode)

#### Usage Example:

```python
from src.news_binary_classifier import NewsBinaryClassifier
import pandas as pd

# 1. Initialize Classifier (Loads models once)
classifier = NewsBinaryClassifier(model_path='models/')

# 2. Load Data
df = pd.read_csv('news_articles.csv')

# 3. Classify (Full Text)
results = classifier.predict(df)

# 4. Classify (Headline-only)
results_fast = classifier.predict_from_headlines(df)

# 5. Get Confidence Scores
results_proba = classifier.predict_with_probability(df)
```

### 2. Incident Intelligence Dashboard

A Streamlit application for interactive analysis, RSS feed collection, and geospatial visualization.

#### Usage:

```bash
streamlit run src/news_intelligence.py
```

---

## Workflows & Recipes

### 🏭 Batch Classification of News

To process a large dataset of news articles efficiently:

```python
from src.news_binary_classifier import classify_news_batch
import pandas as pd

df = pd.read_csv('your_batch_news.csv')

# Returns DataFrame with 'prediction' column
results = classify_news_batch(
    df, 
    model_path='models/', 
    use_headlines_only=False
)
```

### 🚨 Filter Incidents Only

To quickly extract incident-related content from a dataset:

```python
from src.news_binary_classifier import filter_incidents
import pandas as pd

df = pd.read_csv('your_batch_news.csv')

# Returns DataFrame containing ONLY rows with 'prediction' == 'incident'
incidents_df = filter_incidents(df, model_path='models/')
```
