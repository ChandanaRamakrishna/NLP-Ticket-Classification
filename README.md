# 🎫 AI-Driven Ticket Classification & Prioritization

> NLP-Based Classification with TOPSIS Multi-Criteria Prioritization


🚀 **Live App:** [Click here to open](https://chandanaramakrishna-nlp-ticket-classification-app-vop7q1.streamlit.app/)

---

## 📌 Overview

An end-to-end customer support ticket management prototype that:
- **Classifies** complaints into 7 categories using NLP (TF-IDF + Logistic Regression, compared against Random Forest, Naive Bayes and Decision Tree)
- **Discovers topics** in unlabelled complaints using NMF (Non-Negative Matrix Factorization)
- **Prioritizes** tickets using TOPSIS multi-criteria decision making, with adjustable weights
- **Live predictor**: classify a new complaint and get a priority score in real time

---

## 🔄 Project Pipeline

```
JSON Data → Drop empty complaints → Text cleaning → TF-IDF Features
    → NMF Topic Modeling → Classification Models → TOPSIS Ranking
```

---

## 🧪 Models & Results

Trained on **21,072 CFPB complaints** with text (16,854 train / 4,214 held-out test).

| Model | Weighted F1 |
|---|---|
| **Logistic Regression** | **0.80** (macro F1 0.71) |
| Random Forest | 0.77 |
| Naive Bayes | 0.72 |
| Decision Tree | 0.68 |

### Label consolidation
The raw data has 17 CFPB product labels, many of which overlap because CFPB renamed products over time (for example "Credit card" and "Credit card or prepaid card"). With the raw labels, Logistic Regression scored **0.64** weighted F1. Overlapping labels were merged into 7 categories:

| Category | Raw CFPB labels merged |
|---|---|
| Credit/Prepaid card | Credit card; Credit card or prepaid card; Prepaid card |
| Bank account | Bank account or service; Checking or savings account |
| Credit reporting | Credit reporting; Credit reporting, credit repair services, or other personal consumer reports |
| Mortgage | Mortgage |
| Debt collection | Debt collection |
| Other loans | Consumer loan; Student loan; Vehicle loan or lease; Payday, title, or personal loan |
| Other | Money transfers and remaining labels |

Note that merging labels also makes the task easier, so the 0.64 and 0.80 are not a like-for-like model comparison. Small classes (Other, Debt collection, Other loans) score lower, which is why macro F1 (0.71) is below weighted F1 (0.80).

---

## 🎯 TOPSIS Criteria

| Criterion | Direction | Default weight |
|---|---|---|
| Severity | Higher = Better | 30% |
| Resolution Time | Lower = Better | 20% |
| Customer Sentiment | Higher = Better | 20% |
| Priority Level | Higher = Better | 30% |

> ⚠️ The CFPB data does not contain severity, resolution time or priority. In this demo those three criteria are **simulated (random)** to show how the ranking works. Customer Sentiment is computed from the complaint text with TextBlob. Weights are adjustable in the app and are normalized to sum to 1.

---

## 🚀 Run Locally

```bash
# 1. Clone the repo
git clone https://github.com/ChandanaRamakrishna/NLP-Ticket-Classification.git
cd NLP-Ticket-Classification

# 2. Install dependencies
pip install -r requirements.txt

# 3. Run the app
streamlit run app.py
```

The app opens with a small **synthetic demo dataset**. For real-data charts (EDA, topics, TOPSIS), choose **Upload my file** in the sidebar and upload `complaints.json`. The Classification tab and Live Predictor always use the saved model in `artifacts/`, which was trained on the full cleaned data.

---

## 📂 Dataset Format

The app expects a JSON file (CFPB complaints format) with these key fields:

| Field | Description |
|---|---|
| `_source.complaint_what_happened` | The complaint text |
| `_source.product` | Product/category label |

About 73% of rows in the source file have an empty complaint text; these are dropped before training.

Dataset source: [CFPB Consumer Complaints on Kaggle](https://www.kaggle.com/datasets/cfpb/us-consumer-finance-complaints)

---

## 🛠️ Tech Stack

- **Language:** Python 3.10+
- **App:** Streamlit
- **NLP / ML:** scikit-learn (TF-IDF, Logistic Regression, Decision Tree, Naive Bayes, Random Forest), TextBlob, WordCloud
- **Topic Modeling:** NMF (Non-Negative Matrix Factorization)
- **Prioritization:** TOPSIS (custom NumPy implementation)
- **Visualization:** Matplotlib, Seaborn


## 📁 Project Structure

```
NLP-Ticket-Classification/
├── app.py                          # Streamlit application
├── requirements.txt                # Python dependencies (scikit-learn version pinned)
├── README.md                       # This file
└── artifacts/                      # Saved from the training notebook
    ├── best_model.joblib           # TF-IDF + Logistic Regression pipeline
    ├── metrics.json                # Model scores and test-set sizes
    ├── classification_report.csv   # Per-class precision / recall / F1
    └── confusion_matrix.png        # Confusion matrix on the test set
```

---

## ⚠️ Limitations

- Severity, resolution time and priority are simulated; a real deployment would take them from a ticketing system.
- Complaints come from a single bank's consumer complaints, so the model may not transfer to other companies or ticket types.
- Smaller categories have lower F1; no class-imbalance handling beyond label merging was applied to the saved model.
