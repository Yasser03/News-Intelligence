# News Intelligence ⚡

**Scalable News Analysis & Incident Visualizer**

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![Streamlit](https://img.shields.io/badge/UI-Streamlit-FF4B4B.svg)](https://streamlit.io/)
[![LangChain](https://img.shields.io/badge/Agent-LangChain-green.svg)](https://langchain.com/)

News Intelligence is a highly scalable hybrid AI system combining vectorized ML for high-throughput stream filtering and LLM reasoning (Groq/Ollama) for contextual analysis. It auto-discovers RSS feeds, extracts structured event insights, geocodes addresses automatically, and renders interactive maps for real-time security intelligence.

```python
# Fast batch classification is fully vectorized
from src.news_binary_classifier import NewsBinaryClassifier

classifier = NewsBinaryClassifier(model_path='models/')
df_classified = classifier.predict(news_df)
# df_classified['prediction'] contains 'incident' or 'non-incident'
```

---

## 🌟 Why News Intelligence?

### **Hybrid Architecture**
- ⚡ **ML Engine**: Logistic Regression for high-throughput, low-latency filtering (processes 10,000+ items quickly).
- 🧠 **LLM Brain**: Deep reasoning (Groq/Ollama) generates structured impact analysis and contextual summaries for filtered incidents only.

### **Key Features**
- 🔗 **RSS/Atom Feed Discovery** - Auto-discovers and scrapes relevant news articles in parallel via Trafilatura.
- 🗺️ **Geospatial View** - Automatic address/site extraction with geocoding with Folium interactive maps.
- 🔒 **Parallel Processing** - Multi-threaded scraping and LLM invocation for rapid datasets updates.
- 🐼 **LLM Caching** - Built-in LRU cache for LLM queries saves tokens and boosts pipeline speed.
- 🛡️ **Multi-backend Support** - Run completely free with Ollama (local) or scale via Groq (cloud) APIs.

---

## 🚀 Quick Start

### 1. Installation

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration
To use the Groq cloud backend, create a `.env` file in the root directory and add your API key:

```env
GROQ_API_KEY="your_api_key_here"
```

You can generate a new Groq API key from the [Groq Console](https://console.groq.com/keys).

### 3. Model Setup
Ensure your trained model weights exist in the `models/` directory:
- `Logistic Regression_62k.pkl`
- `label_encoder_transformer.pkl`
- `word_vectorizer.pkl`

### 4. Launch the Dashboard
Run the main Streamlit application:

```bash
streamlit run src/news_intilligence.py
```

---

## 📚 Core Components

### 1. News Binary Classifier
The `src/news_binary_classifier.py` provides optimized text vectorization for separating relevant incidents from general news headlines.

| Method | Speed | Richness | Use Case |
|---|---|---|---|
| `predict(df)` | Fast | High | Balance of accuracy and throughput (Title + Desc) |
| `predict_from_headlines(df)` | 🚀 Fastest | Med | Speed is critical (Title only) |
| `predict_with_probability(df)` | Fast | High | Thresholding by confidence |

**Usage snippet:**
```python
from src.news_binary_classifier import classify_news_batch

# Returns DataFrame with 'prediction' column
results = classify_news_batch(
    df, 
    model_path='models/', 
    use_headlines_only=False
)
```

### 2. Incident Analyzer (LLM Pipeline)
Once filtered, articles pass into an LLM analysis chain that parses structured insights back in a single inference call:
- **Incident Type**: Criminality, drugs, protest, bombing, maritime, etc.
- **Impact Level**: `High` (fatalities), `Medium` (injuries), `Low` (minor).
- **Locations**: Extracts country, city, address/landmark automatically.
- **Summary**: Concise five-sentence factual reporting.

### 3. Geospatial Visualizer
Locations extracted are run through `Nominatim` rates-limited geocoding services to populate interactive Map Circles using `folium` right in the dashboard view.

---

## 🔥 Performance & Scaling

- **Speed**: TQDM overhead was removed intentionally for vectorized text cleanup.
- **Rate-Limiting**: Parallel processing uses safe fallback wrappers supporting rate limits for cloud API vendors perfectly.
- **Token Efficiency**: Cached responses based on hash strings speed up re-scraping the same article titles drastically.

---

## 🎓 Learning Resources
- Check out `News_Intelligence_Notebook.ipynb` for step-by-step pipeline creation.
- Check `SKILL.md` for fast recipes setup workflow code.

---

## 🤝 Contributing
Issues and feature suggestions are highly welcomed.

---

## 📜 License
MIT License.

---

## 👨‍💻 Author
**Dr. Yasser Mustafa**

*AI & Data Science Specialist | Theoretical Physics PhD*

- 🎓 PhD in Theoretical Nuclear Physics
- 💼 10+ years in production AI/ML systems
- 📍 Based in Newcastle Upon Tyne, UK
- ✉️ yasser.mustafan@gmail.com
- 🔗 [LinkedIn](https://www.linkedin.com/in/yasser-mustafa) | [GitHub](https://github.com/ymustafa)

---

**Built with ❤️ for rapid threat intelligence situational awareness**

---
