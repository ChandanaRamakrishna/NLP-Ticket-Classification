import os
import re
import json
import string
import warnings

import numpy as np
import pandas as pd
import streamlit as st

warnings.filterwarnings("ignore")

COMPLAINT_COL = "_source.complaint_what_happened"
PRODUCT_COL = "_source.product"
ARTIFACT_DIR = "artifacts"

# ─── Page Config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Ticket Manager",
    page_icon="🎫",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Custom CSS ───────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stTabs [data-baseweb="tab-list"] { gap: 8px; }
    .stTabs [data-baseweb="tab"] {
        background-color: #1e2130;
        border-radius: 8px 8px 0 0;
        padding: 8px 20px;
        color: #a0aec0;
    }
    .stTabs [aria-selected="true"] {
        background-color: #6c63ff !important;
        color: white !important;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #252a3a);
        border: 1px solid #2d3250;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #6c63ff; }
    .metric-label { font-size: 0.85rem; color: #a0aec0; margin-top: 4px; }
    .ticket-card {
        background: #1e2130;
        border-left: 4px solid #6c63ff;
        border-radius: 8px;
        padding: 16px;
        margin: 8px 0;
    }
    .section-header {
        font-size: 1.4rem; font-weight: 700;
        color: #e2e8f0; margin-bottom: 16px;
        border-bottom: 2px solid #6c63ff;
        padding-bottom: 8px;
    }
    div[data-testid="stSidebar"] { background-color: #1a1d2e; }
</style>
""", unsafe_allow_html=True)


# ─── Data helpers ─────────────────────────────────────────────────────────────
@st.cache_data
def get_sample_data():
    """Small synthetic demo dataset (17 template complaints repeated)."""
    complaints = [
        ("I lost my credit card and need a replacement immediately.", "Credit card or prepaid card"),
        ("Unauthorized charges appeared on my account. Fraud!", "Credit card or prepaid card"),
        ("My credit card was declined at the store.", "Credit card or prepaid card"),
        ("They charged me twice for the same transaction.", "Credit card or prepaid card"),
        ("My account was locked without any prior notice.", "Checking or savings account"),
        ("The bank deducted money from my savings without explanation.", "Checking or savings account"),
        ("I cannot access my online banking portal since yesterday.", "Checking or savings account"),
        ("Wrong balance shown in my checking account.", "Checking or savings account"),
        ("Someone stole my debit card and made purchases.", "Theft/Dispute reporting"),
        ("I am disputing a charge of $500 made without my consent.", "Theft/Dispute reporting"),
        ("Identity theft - someone opened an account in my name.", "Theft/Dispute reporting"),
        ("My loan application has been pending for 3 months.", "Mortgage/Loan"),
        ("The mortgage interest rate changed without notification.", "Mortgage/Loan"),
        ("I was denied a home loan with no reason given.", "Mortgage/Loan"),
        ("The customer service is extremely slow and unhelpful.", "Others"),
        ("I have been waiting 2 weeks for a response.", "Others"),
        ("The mobile app keeps crashing.", "Others"),
    ] * 60

    texts, labels = zip(*complaints)
    n = len(texts)
    df = pd.DataFrame({
        COMPLAINT_COL: list(texts),
        PRODUCT_COL: list(labels),
        "_source.date_received": pd.date_range("2023-01-01", periods=n, freq="h"),
    })
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


def drop_empty(df):
    """Remove rows whose complaint text is missing or blank (BEFORE sampling)."""
    if COMPLAINT_COL not in df.columns:
        return df
    s = df[COMPLAINT_COL].astype(str).str.strip()
    return df[~s.isin(["", "nan", "None"])].reset_index(drop=True)


def consolidate_product(p):
    """Merge near-duplicate CFPB product labels (CFPB renamed several products,
    so the same kind of complaint appears under two names). Must match the
    mapping used in the notebook export cell."""
    p = str(p).lower()
    if "credit reporting" in p:
        return "Credit reporting"
    if "credit card" in p or "prepaid" in p:
        return "Credit/Prepaid card"
    if "checking" in p or "bank account" in p:
        return "Bank account"
    if "mortgage" in p:
        return "Mortgage"
    if "debt collection" in p:
        return "Debt collection"
    if "loan" in p:
        return "Other loans"
    return "Other"


def clean_text(text):
    text = str(text).lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[%s]' % re.escape(string.punctuation), '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text


def notebook_clean(text):
    """The exported model's TF-IDF step removes stopwords itself, so only the
    basic cleaning used during training (clean_text) is applied here."""
    return clean_text(text)


STOPWORDS = {
    "i","me","my","myself","we","our","ours","ourselves","you","your","yours",
    "he","him","his","she","her","hers","it","its","they","them","their",
    "what","which","who","this","that","these","those","am","is","are","was",
    "were","be","been","being","have","has","had","do","does","did","will",
    "would","could","should","may","might","shall","can","a","an","the","and",
    "but","if","or","as","at","by","for","of","on","to","in","with","about",
    "from","into","through","during","before","after","up","down","out","off",
    "over","under","then","once","here","there","when","where","why","how",
    "all","both","each","more","most","other","some","such","no","not","only",
    "same","so","than","too","very","just","don","now","xx","xxxx","x",
}


def remove_stopwords(text):
    return ' '.join(w for w in text.split() if w not in STOPWORDS)


def preprocess(text):
    return remove_stopwords(clean_text(text))


@st.cache_data
def preprocess_df(df):
    df = df.copy()
    df[COMPLAINT_COL] = df[COMPLAINT_COL].astype(str).str.strip()
    df["clean_text"] = df[COMPLAINT_COL].apply(preprocess)
    # drop rows that are empty after cleaning as well
    df = df[df["clean_text"].str.len() > 0].reset_index(drop=True)
    return df


# ─── Saved notebook artifacts ────────────────────────────────────────────────
@st.cache_resource
def load_artifacts():
    """Load the model + metrics exported from the notebook. Returns None if absent."""
    try:
        import joblib
        model = joblib.load(os.path.join(ARTIFACT_DIR, "best_model.joblib"))
        with open(os.path.join(ARTIFACT_DIR, "metrics.json")) as f:
            metrics = json.load(f)
        report_df = pd.read_csv(os.path.join(ARTIFACT_DIR, "classification_report.csv"), index_col=0)
        cm_path = os.path.join(ARTIFACT_DIR, "confusion_matrix.png")
        return {
            "model": model,
            "metrics": metrics,
            "report": report_df,
            "cm_path": cm_path if os.path.exists(cm_path) else None,
        }
    except Exception:
        return None


# ─── Demo training (on the current sample only) ──────────────────────────────
@st.cache_resource
def train_models(df):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import train_test_split
    from sklearn.metrics import f1_score, classification_report

    X = df['clean_text']
    y = df[PRODUCT_COL]

    vectorizer = TfidfVectorizer(max_features=3000)
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42
    )

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000, class_weight="balanced"),
        "Decision Tree": DecisionTreeClassifier(random_state=42, class_weight="balanced"),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42,
                                                n_jobs=-1, class_weight="balanced"),
    }

    results = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted', zero_division=0)
        report = classification_report(y_test, y_pred, output_dict=True, zero_division=0)
        results[name] = {
            "model": clf, "f1": f1,
            "y_test": y_test, "y_pred": y_pred,
            "report": report,
        }

    return vectorizer, results, X_train, X_test, y_train, y_test


@st.cache_resource
def run_nmf(df, n_topics=5):
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import NMF

    df = df.copy()
    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=2000, stop_words='english')
    tfidf = vectorizer.fit_transform(df['clean_text'])
    nmf = NMF(n_components=n_topics, random_state=42, max_iter=400)
    W = nmf.fit_transform(tfidf)

    feature_names = vectorizer.get_feature_names_out()
    topics = {}
    for i, comp in enumerate(nmf.components_):
        top_words = [feature_names[j] for j in comp.argsort()[:-11:-1]]
        # Neutral names: topic labels should be assigned by inspecting the words,
        # not hardcoded by index (NMF topic order is arbitrary).
        topics[f"Topic {i + 1}"] = top_words

    df['topic'] = W.argmax(axis=1)
    return topics, df


def compute_topsis(df_t):
    criteria = ['Severity', 'Resolution Time', 'Customer Sentiment', 'Priority']
    matrix = df_t[criteria].copy().astype(float)
    matrix['Resolution Time'] = matrix['Resolution Time'].max() - matrix['Resolution Time']
    norms = np.sqrt((matrix ** 2).sum())
    norm_matrix = matrix / norms
    weights = np.array([0.30, 0.20, 0.20, 0.30])
    weighted = norm_matrix * weights
    ideal_best = weighted.max()
    ideal_worst = weighted.min()
    d_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    d_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))
    return d_worst / (d_best + d_worst)


def sentiment_01(text):
    """TextBlob polarity rescaled from [-1, 1] to [0, 1]; 0.5 if TextBlob missing."""
    try:
        from textblob import TextBlob
        return (TextBlob(str(text)).sentiment.polarity + 1) / 2
    except ImportError:
        return 0.5


# ─── Classification tab renderers ────────────────────────────────────────────
def render_saved_results(art):
    import matplotlib.pyplot as plt

    m = art["metrics"]
    st.markdown("#### 🏆 Model F1-Score Comparison")
    scores = pd.Series(m["models"]).sort_values(ascending=False)

    fig, ax = plt.subplots(figsize=(7, 3), facecolor='#1e2130')
    ax.set_facecolor('#1e2130')
    colors_bar = ['#6c63ff', '#4fd1c5', '#f6ad55', '#fc8181']
    bars = ax.barh(scores.index[::-1], scores.values[::-1],
                   color=colors_bar[:len(scores)][::-1], edgecolor='#1e2130', alpha=0.9)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel(m.get("metric", "F1-Score"), color='white')
    ax.set_title("Model Comparison", color='white')
    ax.tick_params(colors='white')
    for bar, val in zip(bars, scores.values[::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va='center', color='white', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    n_train = f"{m['n_train']:,}" if "n_train" in m else "?"
    n_test = f"{m['n_test']:,}" if "n_test" in m else "?"
    extra = ""
    if "macro_f1" in m:
        extra += f" Macro F1 of best model: {m['macro_f1']:.2f}."
    if "baseline_raw_label_f1" in m:
        extra += (f" Before merging near-duplicate labels ({m.get('n_raw_classes', '?')} raw classes), "
                  f"Logistic Regression scored {m['baseline_raw_label_f1']:.2f}.")
    st.caption(
        f"{m.get('metric', 'F1')} on a held-out test set. "
        f"Dataset: {m.get('dataset', 'CFPB complaints')} — "
        f"{n_train} training / {n_test} test complaints (full notebook training run).{extra}"
    )
    st.success(f"🏆 Best model: **{scores.index[0]}** — {m.get('metric', 'F1')}: **{scores.iloc[0]:.2f}**")

    if art["cm_path"]:
        st.markdown("#### 🔢 Confusion Matrix — Best Model")
        st.image(art["cm_path"])

    st.markdown("#### 📋 Classification Report — Best Model")
    st.dataframe(art["report"].round(3), use_container_width=True)


def render_retrain_demo(df, is_synthetic):
    import matplotlib.pyplot as plt
    import seaborn as sns
    from sklearn.metrics import confusion_matrix

    if PRODUCT_COL not in df.columns:
        st.warning(f"Target column `{PRODUCT_COL}` not found.")
        return

    st.caption(
        f"Demo only: models trained on a {len(df):,}-ticket sample using "
        f"`{PRODUCT_COL}` labels ({df[PRODUCT_COL].nunique()} classes). "
        "These are NOT the reported project results."
    )
    if is_synthetic:
        st.warning(
            "The built-in sample data is 17 template complaints repeated many times, so "
            "train and test sets contain near-identical rows and scores are inflated. "
            "Do not quote these numbers."
        )

    with st.spinner("Training models… (this may take a moment)"):
        vectorizer, model_results, X_train, X_test, y_train, y_test = train_models(df)

    f1_df = pd.DataFrame([(n, r['f1']) for n, r in model_results.items()],
                         columns=['Model', 'F1-Score']).sort_values('F1-Score', ascending=False).reset_index(drop=True)

    st.markdown("#### 🏆 Model F1-Score Comparison (demo sample)")
    fig, ax = plt.subplots(figsize=(7, 3), facecolor='#1e2130')
    ax.set_facecolor('#1e2130')
    colors_bar = ['#6c63ff', '#4fd1c5', '#f6ad55', '#fc8181']
    bars = ax.barh(f1_df['Model'][::-1], f1_df['F1-Score'][::-1],
                   color=colors_bar[::-1], edgecolor='#1e2130', alpha=0.9)
    ax.set_xlim(0, 1.1)
    ax.set_xlabel("F1-Score (weighted)", color='white')
    ax.set_title("Model Comparison", color='white')
    ax.tick_params(colors='white')
    for bar, val in zip(bars, f1_df['F1-Score'][::-1]):
        ax.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va='center', color='white', fontsize=10)
    plt.tight_layout()
    st.pyplot(fig)
    plt.close()

    best_name = f1_df.iloc[0]['Model']
    best_res = model_results[best_name]

    # Confusion matrix: use only labels present in y_test / y_pred so axes line up
    labels = sorted(set(best_res['y_test']) | set(best_res['y_pred']))
    cm = confusion_matrix(best_res['y_test'], best_res['y_pred'], labels=labels)
    fig2, ax2 = plt.subplots(figsize=(7, 5), facecolor='#1e2130')
    ax2.set_facecolor('#1e2130')
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=labels, yticklabels=labels, ax=ax2, cbar_kws={'shrink': 0.8})
    ax2.set_xlabel("Predicted", color='white')
    ax2.set_ylabel("Actual", color='white')
    ax2.set_title(f"Confusion Matrix — {best_name} (demo sample)", color='white')
    ax2.tick_params(colors='white', labelsize=7)
    plt.xticks(rotation=30, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    st.pyplot(fig2)
    plt.close()

    rows = []
    for cls, mt in best_res['report'].items():
        if isinstance(mt, dict):
            rows.append({'Class': cls, 'Precision': round(mt['precision'], 3),
                         'Recall': round(mt['recall'], 3),
                         'F1-Score': round(mt['f1-score'], 3),
                         'Support': int(mt['support'])})
    st.dataframe(pd.DataFrame(rows), use_container_width=True)


# ─── Sidebar ──────────────────────────────────────────────────────────────────
is_synthetic = False
with st.sidebar:
    st.markdown("## 🎫 AI Ticket Manager")
    st.markdown("NLP-Based Classification & TOPSIS Prioritization")
    st.divider()

    st.markdown("### 📂 Data Source")
    data_mode = st.radio("Choose data source:", ["Use sample data", "Upload my file"])

    df_raw = None
    if data_mode == "Upload my file":
        uploaded = st.file_uploader(
            "Upload your file",
            type=["json", "csv", "xlsx", "xls"],
            help=f"Needs columns `{COMPLAINT_COL}` and `{PRODUCT_COL}`"
        )
        if uploaded:
            try:
                name = uploaded.name.lower()
                if name.endswith(".json"):
                    raw = json.loads(uploaded.read().decode("utf-8", errors="ignore"))
                    df_raw = pd.json_normalize(raw)
                elif name.endswith(".csv"):
                    df_raw = pd.read_csv(uploaded)
                else:
                    df_raw = pd.read_excel(uploaded)

                if COMPLAINT_COL not in df_raw.columns:
                    st.error(f"Missing column `{COMPLAINT_COL}`.")
                    df_raw = None
                else:
                    st.success(
                        f"✅ Loaded {len(df_raw):,} records "
                        f"({len(drop_empty(df_raw)):,} with complaint text)"
                    )
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        df_raw = get_sample_data()
        is_synthetic = True
        st.info(f"📊 Using {len(df_raw):,} synthetic demo records")

    st.divider()
    st.markdown("### ⚙️ Settings")
    n_topics = st.slider("NMF Topics", 3, 8, 5)
    n_sample = st.slider("Demo sample size (for speed)", 200, 5000, 1500, step=100)

    st.divider()
    st.markdown("**Made by:** Ananya PA & Chandana R")
    st.markdown("**Guide:** Ms. Smriti Kumari")

# ─── Main App ─────────────────────────────────────────────────────────────────
st.title("🎫 AI-Driven Ticket Classification & Prioritization")
st.markdown(
    "Automates customer support ticket **classification** using NLP and "
    "**prioritization** using TOPSIS multi-criteria decision making."
)

if df_raw is None:
    st.warning("👈 Please select or upload data from the sidebar to get started.")
    st.stop()

# 1) drop empty complaints first, 2) then sample, 3) then clean text
df_usable = drop_empty(df_raw)
n_available = len(df_usable)
if n_available == 0:
    st.error("No complaints with text were found in this file.")
    st.stop()

if n_available > n_sample:
    df_sample = df_usable.sample(n_sample, random_state=42).reset_index(drop=True)
else:
    df_sample = df_usable.copy()

df = preprocess_df(df_sample)
if not is_synthetic and PRODUCT_COL in df.columns:
    df[PRODUCT_COL] = df[PRODUCT_COL].apply(consolidate_product)
artifacts = load_artifacts()

# ─── Tabs ─────────────────────────────────────────────────────────────────────
tabs = st.tabs([
    "🏠 Overview",
    "🔍 EDA",
    "🧠 Topic Modeling",
    "🤖 Classification",
    "🎯 TOPSIS Prioritization",
    "🔮 Live Predictor",
])

# ══════════════════════════════════════════════════════════════════════════════
# TAB 1: OVERVIEW
# ══════════════════════════════════════════════════════════════════════════════
with tabs[0]:
    st.markdown('<div class="section-header">Project Overview</div>', unsafe_allow_html=True)

    col1, col2, col3, col4 = st.columns(4)
    total = len(df)
    cats = df[PRODUCT_COL].nunique() if PRODUCT_COL in df.columns else "—"

    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{total:,}</div><div class="metric-label">Tickets in demo sample</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{cats}</div><div class="metric-label">Categories in sample</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown('<div class="metric-card"><div class="metric-value">4</div><div class="metric-label">ML Models</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown('<div class="metric-card"><div class="metric-value">TOPSIS</div><div class="metric-label">Prioritization</div></div>', unsafe_allow_html=True)

    st.caption(
        f"Demo sample of {total:,} tickets drawn from {n_available:,} complaints with usable text. "
        "Headline model results on the Classification tab come from the full notebook training run."
    )

    st.markdown("---")
    st.markdown("### 🔄 Project Pipeline")
    pipeline_steps = [
        ("1️⃣", "Data Loading", "JSON complaints → Pandas DataFrame via json_normalize"),
        ("2️⃣", "Text Preprocessing", "Drop empty complaints → Clean → Remove stopwords"),
        ("3️⃣", "EDA", "Category distribution, word frequencies, ticket lengths"),
        ("4️⃣", "Feature Extraction", "TF-IDF vectorization → sparse feature matrix"),
        ("5️⃣", "Topic Modeling", "NMF groups unlabelled tickets into clusters"),
        ("6️⃣", "Classification", "Logistic Regression / Decision Tree / Naive Bayes / Random Forest"),
        ("7️⃣", "TOPSIS Ranking", "Multi-criteria ranking by Severity · Sentiment · Resolution Time · Priority"),
    ]
    for icon, title, desc in pipeline_steps:
        st.markdown(f"**{icon} {title}** — {desc}")

    st.markdown("---")
    st.markdown("### 📋 Raw Data Preview")
    cols_show = [c for c in [PRODUCT_COL, COMPLAINT_COL, 'clean_text'] if c in df.columns]
    st.dataframe(df[cols_show].head(10), use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 2: EDA
# ══════════════════════════════════════════════════════════════════════════════
with tabs[1]:
    import matplotlib.pyplot as plt
    import matplotlib
    matplotlib.rcParams.update({'text.color': 'white', 'axes.labelcolor': 'white',
                                'xtick.color': 'white', 'ytick.color': 'white',
                                'axes.facecolor': '#1e2130', 'figure.facecolor': '#1e2130',
                                'axes.edgecolor': '#2d3250', 'grid.color': '#2d3250'})

    st.markdown('<div class="section-header">Exploratory Data Analysis</div>', unsafe_allow_html=True)

    if PRODUCT_COL not in df.columns:
        st.warning("No product/category column found in uploaded data.")
    else:
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### 📊 Ticket Category Distribution")
            cat_counts = df[PRODUCT_COL].value_counts()
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = ['#6c63ff', '#f093fb', '#4fd1c5', '#f6ad55', '#fc8181']
            bar_colors = [colors[i % len(colors)] for i in range(len(cat_counts))]
            bars = ax.barh(cat_counts.index[::-1], cat_counts.values[::-1], color=bar_colors[::-1])
            ax.set_xlabel("Number of Tickets")
            ax.set_title("Complaint Categories", color='white', fontsize=12)
            for bar, val in zip(bars, cat_counts.values[::-1]):
                ax.text(bar.get_width() + 1, bar.get_y() + bar.get_height() / 2,
                        str(val), va='center', color='white', fontsize=9)
            plt.tight_layout()
            st.pyplot(fig)
            plt.close()

        with col_right:
            st.markdown("#### 🥧 Category Share")
            fig2, ax2 = plt.subplots(figsize=(5, 4))
            wedge_colors = ['#6c63ff', '#f093fb', '#4fd1c5', '#f6ad55', '#fc8181', '#68d391']
            ax2.pie(cat_counts.values, labels=cat_counts.index,
                    autopct='%1.1f%%',
                    colors=[wedge_colors[i % len(wedge_colors)] for i in range(len(cat_counts))],
                    textprops={'color': 'white', 'fontsize': 8},
                    startangle=140, pctdistance=0.8)
            ax2.set_title("Distribution of Categories", color='white')
            plt.tight_layout()
            st.pyplot(fig2)
            plt.close()

        st.markdown("#### ☁️ Word Cloud — Most Frequent Terms")
        try:
            from wordcloud import WordCloud
            all_text = ' '.join(df['clean_text'].dropna())
            wc = WordCloud(width=900, height=350, background_color='#1e2130',
                           colormap='cool', max_words=120).generate(all_text)
            fig3, ax3 = plt.subplots(figsize=(10, 3.5))
            ax3.imshow(wc, interpolation='bilinear')
            ax3.axis('off')
            st.pyplot(fig3)
            plt.close()
        except ImportError:
            st.info("Install `wordcloud` to see the word cloud: `pip install wordcloud`")

        st.markdown("#### 📏 Ticket Length Distribution")
        df['text_len'] = df['clean_text'].apply(lambda x: len(str(x).split()))
        fig4, ax4 = plt.subplots(figsize=(8, 3))
        ax4.hist(df['text_len'], bins=30, color='#6c63ff', edgecolor='#1e2130', alpha=0.85)
        ax4.set_xlabel("Number of Words")
        ax4.set_ylabel("Frequency")
        ax4.set_title("Distribution of Ticket Lengths", color='white')
        plt.tight_layout()
        st.pyplot(fig4)
        plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 3: TOPIC MODELING
# ══════════════════════════════════════════════════════════════════════════════
with tabs[2]:
    st.markdown('<div class="section-header">Topic Modeling with NMF</div>', unsafe_allow_html=True)
    st.markdown(
        "Non-Negative Matrix Factorization (NMF) discovers hidden topics in unlabelled "
        "complaints by decomposing the TF-IDF matrix into interpretable components. "
        "Topics are named by their top words; interpret them to assign business labels."
    )

    with st.spinner("Running NMF topic modeling…"):
        topics, df_topics = run_nmf(df, n_topics)

    cols = st.columns(min(3, len(topics)))
    topic_colors = ['#6c63ff', '#f093fb', '#4fd1c5', '#f6ad55', '#fc8181', '#68d391', '#76e4f7', '#fbb6ce']
    for i, (topic_name, words) in enumerate(topics.items()):
        c = cols[i % len(cols)]
        color = topic_colors[i % len(topic_colors)]
        c.markdown(
            f"<div style='background:#1e2130;border-left:4px solid {color};"
            f"border-radius:8px;padding:14px;margin:8px 0'>"
            f"<b style='color:{color}'>{topic_name}</b><br/>"
            f"<span style='color:#a0aec0;font-size:0.88rem'>{' · '.join(words[:8])}</span>"
            f"</div>",
            unsafe_allow_html=True,
        )

    st.markdown("---")
    st.markdown("#### 📊 Topic Distribution in Dataset")
    import matplotlib.pyplot as plt
    topic_names_list = list(topics.keys())
    topic_dist = df_topics['topic'].value_counts().sort_index()
    labels = [topic_names_list[i] if i < len(topic_names_list) else f"Topic {i + 1}" for i in topic_dist.index]
    fig5, ax5 = plt.subplots(figsize=(8, 3.5), facecolor='#1e2130')
    ax5.set_facecolor('#1e2130')
    ax5.bar(labels, topic_dist.values,
            color=topic_colors[:len(labels)], edgecolor='#1e2130', alpha=0.9)
    ax5.set_ylabel("Count", color='white')
    ax5.set_title("Tickets per Topic", color='white')
    ax5.tick_params(colors='white')
    plt.xticks(rotation=20, ha='right', color='white', fontsize=8)
    plt.tight_layout()
    st.pyplot(fig5)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 4: CLASSIFICATION
# ══════════════════════════════════════════════════════════════════════════════
with tabs[3]:
    st.markdown('<div class="section-header">Supervised Classification Models</div>', unsafe_allow_html=True)

    if artifacts is not None:
        render_saved_results(artifacts)
        st.markdown("---")
        with st.expander("🧪 Optional: retrain quickly on the current demo sample"):
            if st.button("Train on current sample"):
                st.session_state["retrain_demo"] = True
            if st.session_state.get("retrain_demo"):
                render_retrain_demo(df, is_synthetic)
    else:
        st.warning(
            "No saved model found in `artifacts/`. Showing a quick demo trained on the current "
            "sample instead. Export the notebook model to show the real results."
        )
        render_retrain_demo(df, is_synthetic)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: TOPSIS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-header">TOPSIS-Based Ticket Prioritization</div>', unsafe_allow_html=True)
    st.markdown(
        "TOPSIS ranks tickets by their closeness to the **ideal solution** across "
        "four weighted criteria: Severity, Resolution Time, Sentiment, and Priority."
    )
    st.info(
        "ℹ️ Severity, Resolution Time and Priority are **simulated** (random) for demonstration, "
        "since the CFPB data does not contain them. Customer Sentiment is computed from the "
        "complaint text. Weights are normalized to sum to 1."
    )

    col_w1, col_w2, col_w3, col_w4 = st.columns(4)
    with col_w1: w_sev = st.slider("Severity weight", 0.1, 0.6, 0.30, 0.05)
    with col_w2: w_res = st.slider("Resolution Time weight", 0.1, 0.6, 0.20, 0.05)
    with col_w3: w_sen = st.slider("Sentiment weight", 0.1, 0.6, 0.20, 0.05)
    with col_w4: w_pri = st.slider("Priority weight", 0.1, 0.6, 0.30, 0.05)

    n_t = min(100, len(df))
    df_topsis = df.head(n_t).copy()

    # Sentiment from the RAW complaint text (cleaning strips most polarity signal)
    df_topsis['Customer Sentiment'] = df_topsis[COMPLAINT_COL].apply(sentiment_01)

    rng = np.random.default_rng(1)
    df_topsis['Severity'] = rng.integers(1, 6, size=n_t)
    df_topsis['Resolution Time'] = rng.integers(1, 48, size=n_t)
    df_topsis['Priority'] = rng.integers(1, 5, size=n_t)

    matrix = df_topsis[['Severity', 'Resolution Time', 'Customer Sentiment', 'Priority']].astype(float).copy()
    matrix['Resolution Time'] = matrix['Resolution Time'].max() - matrix['Resolution Time']
    norms = np.sqrt((matrix ** 2).sum())
    norm_matrix = matrix / norms
    w = np.array([w_sev, w_res, w_sen, w_pri])
    weights_arr = w / w.sum()
    weighted = norm_matrix * weights_arr
    ideal_best = weighted.max()
    ideal_worst = weighted.min()
    d_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    d_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))
    df_topsis['TOPSIS Score'] = d_worst / (d_best + d_worst)
    df_topsis['Rank'] = df_topsis['TOPSIS Score'].rank(ascending=False, method='dense').astype(int)

    def priority_label(score):
        if score >= 0.65: return "🔴 High"
        elif score >= 0.45: return "🟡 Medium"
        else: return "🟢 Low"

    df_topsis['Priority Level'] = df_topsis['TOPSIS Score'].apply(priority_label)
    df_sorted = df_topsis.sort_values('TOPSIS Score', ascending=False).head(20)

    st.markdown(f"#### 🏅 Top Prioritized Tickets (showing top 20 of {n_t})")
    display_cols = [COMPLAINT_COL, PRODUCT_COL,
                    'Severity', 'Resolution Time', 'Customer Sentiment',
                    'Priority', 'TOPSIS Score', 'Rank', 'Priority Level']
    display_cols = [c for c in display_cols if c in df_sorted.columns]
    st.dataframe(
        df_sorted[display_cols].rename(columns={
            COMPLAINT_COL: 'Complaint',
            PRODUCT_COL: 'Category',
        }),
        use_container_width=True,
    )

    import matplotlib.pyplot as plt
    st.markdown("#### 📊 TOPSIS Score Distribution")
    fig8, ax8 = plt.subplots(figsize=(8, 3), facecolor='#1e2130')
    ax8.set_facecolor('#1e2130')
    ax8.hist(df_topsis['TOPSIS Score'], bins=20, color='#6c63ff', edgecolor='#1e2130', alpha=0.85)
    ax8.axvline(0.65, color='#fc8181', linestyle='--', label='High priority threshold')
    ax8.axvline(0.45, color='#f6ad55', linestyle='--', label='Medium priority threshold')
    ax8.set_xlabel("TOPSIS Score", color='white')
    ax8.set_ylabel("Frequency", color='white')
    ax8.set_title("Distribution of Priority Scores", color='white')
    ax8.tick_params(colors='white')
    ax8.legend(facecolor='#1e2130', edgecolor='#2d3250', labelcolor='white')
    plt.tight_layout()
    st.pyplot(fig8)
    plt.close()


# ══════════════════════════════════════════════════════════════════════════════
# TAB 6: LIVE PREDICTOR
# ══════════════════════════════════════════════════════════════════════════════
with tabs[5]:
    st.markdown('<div class="section-header">🔮 Live Ticket Predictor</div>', unsafe_allow_html=True)
    st.markdown("Enter a new complaint below to classify it and get a TOPSIS priority score.")

    predict_category = None
    predictor_name = ""

    if artifacts is not None:
        _model = artifacts["model"]
        predictor_name = artifacts["metrics"].get("best_model", "Logistic Regression") + " (notebook-trained)"

        def predict_category(text):
            return _model.predict([notebook_clean(text)])[0]

    elif PRODUCT_COL in df.columns:
        with st.spinner("Training demo model…"):
            _vec, _results, _, _, _, _ = train_models(df)
        _best_name = max(_results, key=lambda k: _results[k]['f1'])
        _best_clf = _results[_best_name]['model']
        predictor_name = f"{_best_name} (demo model trained on sample)"

        def predict_category(text):
            return _best_clf.predict(_vec.transform([preprocess(text)]))[0]

    if predict_category is None:
        st.warning("No trained model available. Add `artifacts/best_model.joblib` or upload a labeled dataset.")
    else:
        examples = [
            "I lost my credit card and need a replacement urgently.",
            "There are unauthorized charges on my account, possible fraud.",
            "My loan application has been pending for over 3 months.",
            "The bank locked my account without any explanation.",
            "Customer service is terrible and no one responds.",
        ]
        example_choice = st.selectbox("Or pick an example:", ["(type your own)"] + examples)
        default_text = "" if example_choice == "(type your own)" else example_choice

        ticket_input = st.text_area(
            "✏️ Complaint Text",
            value=default_text,
            height=120,
            placeholder="Describe the customer issue here…"
        )

        col_sev, col_res, col_pri = st.columns(3)
        with col_sev: sev_input = st.slider("Severity (1–5)", 1, 5, 3)
        with col_res: res_input = st.slider("Est. Resolution Time (hours)", 1, 72, 24)
        with col_pri: pri_input = st.slider("Priority (1–4)", 1, 4, 2)

        if st.button("🚀 Classify & Prioritize", type="primary", use_container_width=True):
            if ticket_input.strip():
                cleaned = preprocess(ticket_input)
                pred_category = predict_category(ticket_input)
                sentiment = sentiment_01(ticket_input)

                single = pd.DataFrame({
                    'Severity': [sev_input, 1, 5],
                    'Resolution Time': [res_input, 1, 72],
                    'Customer Sentiment': [sentiment, 0.0, 1.0],
                    'Priority': [pri_input, 1, 4],
                })
                topsis_score = compute_topsis(single).iloc[0]

                if topsis_score >= 0.65:
                    priority, color = "🔴 HIGH", "#fc8181"
                elif topsis_score >= 0.45:
                    priority, color = "🟡 MEDIUM", "#f6ad55"
                else:
                    priority, color = "🟢 LOW", "#68d391"

                st.markdown("---")
                r1, r2, r3 = st.columns(3)
                r1.markdown(
                    f"<div class='metric-card'><div class='metric-value' style='color:#6c63ff;font-size:1.4rem'>{pred_category}</div>"
                    f"<div class='metric-label'>Predicted Category</div></div>",
                    unsafe_allow_html=True,
                )
                r2.markdown(
                    f"<div class='metric-card'><div class='metric-value' style='color:{color}'>{topsis_score:.3f}</div>"
                    f"<div class='metric-label'>TOPSIS Score</div></div>",
                    unsafe_allow_html=True,
                )
                r3.markdown(
                    f"<div class='metric-card'><div class='metric-value' style='color:{color}'>{priority}</div>"
                    f"<div class='metric-label'>Priority Level</div></div>",
                    unsafe_allow_html=True,
                )

                st.markdown("---")
                mood = 'Positive 😊' if sentiment > 0.55 else 'Negative 😠' if sentiment < 0.45 else 'Neutral 😐'
                st.markdown(
                    f"<div class='ticket-card'>"
                    f"<b style='color:#a0aec0'>Sentiment Score:</b> {sentiment:.3f} ({mood})<br/>"
                    f"<b style='color:#a0aec0'>Classifier Used:</b> {predictor_name}<br/>"
                    f"<b style='color:#a0aec0'>Cleaned Text:</b> {cleaned[:200]}…"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.warning("Please enter a complaint text.")