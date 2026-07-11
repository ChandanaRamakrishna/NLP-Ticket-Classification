# 🎫 AI-Driven Ticket Classification & Prioritization

> NLP-Based Classification with TOPSIS Multi-Criteria Prioritization  
> **By:** Chandana R


🚀 **Live App:** [Click here to open](https://chandanaramakrishna-nlp-ticket-classification-app-vop7q1.streamlit.app/)

---

## 📌 Overview

An end-to-end AI-powered customer support ticket management system that:
- **Classifies** tickets using NLP (TF-IDF + Logistic Regression / Random Forest)
- **Discovers topics** in unlabelled data using NMF (Non-Negative Matrix Factorization)
- **Prioritizes** tickets using TOPSIS multi-criteria decision making
- **Live predictor** — classify any new complaint in real time

---

## 🔄 Project Pipeline

```
JSON Data → Text Preprocessing → EDA → TF-IDF Features
    → NMF Topic Modeling → Classification Models → TOPSIS Ranking
```

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/your-username/nlp-ticket-classification.git
cd nlp-ticket-classification

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app works **without any dataset** — it ships with sample data. You can also upload your own `complaints.json`.

---

## 📂 Dataset Format

The app expects a JSON file (CFPB complaints format) with these key fields:

| Field | Description |
|---|---|
| `_source.complaint_what_happened` | The complaint text |
| `_source.product` | Product/category label |

Dataset source: [CFPB Consumer Complaints on Kaggle](https://www.kaggle.com/datasets/cfpb/us-consumer-finance-complaints)

---

## 🧪 Models & Results

| Model | F1-Score |
|---|---|
| Logistic Regression | 0.95–1.00 |
| Random Forest | ~0.92 |
| Decision Tree | ~0.89 |
| Naive Bayes | ~0.70 |

---

## 🎯 TOPSIS Criteria

| Criterion | Direction | Weight |
|---|---|---|
| Severity | Higher = Better | 30% |
| Resolution Time | Lower = Better | 20% |
| Customer Sentiment | Higher = Better | 20% |
| Priority Level | Higher = Better | 30% |

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **App:** Streamlit
- **NLP:** scikit-learn, TextBlob, WordCloud
- **ML:** Logistic Regression, Decision Tree, Naive Bayes, Random Forest
- **Topic Modeling:** NMF (Non-Negative Matrix Factorization)
- **Prioritization:** TOPSIS (custom implementation)
- **Visualization:** Matplotlib, Seaborn

---

## ☁️ Deploy to Streamlit Cloud (Free)

1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → **New app**
3. Select your repo → `app.py` → Deploy

No dataset upload needed — the app uses built-in sample data.

---

## 📁 Project Structure

```
nlp-ticket-classification/
├── app.py              # Main Streamlit application
├── requirements.txt    # Python dependencies
└── README.md           # This file
```
