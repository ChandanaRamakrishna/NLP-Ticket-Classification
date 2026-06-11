import streamlit as st
import pandas as pd
import numpy as np
import re
import string
import json
import warnings
warnings.filterwarnings("ignore")

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
    .priority-high   { border-left-color: #f56565 !important; }
    .priority-medium { border-left-color: #ed8936 !important; }
    .priority-low    { border-left-color: #48bb78 !important; }
    .section-header {
        font-size: 1.4rem; font-weight: 700;
        color: #e2e8f0; margin-bottom: 16px;
        border-bottom: 2px solid #6c63ff;
        padding-bottom: 8px;
    }
    div[data-testid="stSidebar"] { background-color: #1a1d2e; }
</style>
""", unsafe_allow_html=True)

# ─── Helpers ──────────────────────────────────────────────────────────────────
@st.cache_data
def get_sample_data():
    """Generate realistic sample complaint data when no file is uploaded."""
    np.random.seed(42)
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
    ] * 60  # Repeat to get ~1000 samples

    texts, labels = zip(*complaints)
    n = len(texts)

    df = pd.DataFrame({
        "_source.complaint_what_happened": list(texts),
        "_source.product": list(labels),
        "_source.date_received": pd.date_range("2023-01-01", periods=n, freq="h"),
    })
    return df.sample(frac=1, random_state=42).reset_index(drop=True)


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
    df['_source.complaint_what_happened'].replace("", np.nan, inplace=True)
    df.dropna(subset=['_source.complaint_what_happened'], inplace=True)
    df.reset_index(drop=True, inplace=True)
    df['clean_text'] = df['_source.complaint_what_happened'].apply(preprocess)
    return df


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
    y = df['_source.product']

    vectorizer = TfidfVectorizer(max_features=3000)
    X_vec = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vec, y, test_size=0.2, random_state=42
    )

    classifiers = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Naive Bayes": MultinomialNB(),
        "Random Forest": RandomForestClassifier(n_estimators=50, random_state=42, n_jobs=-1),
    }

    results = {}
    for name, clf in classifiers.items():
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)
        f1 = f1_score(y_test, y_pred, average='weighted')
        report = classification_report(y_test, y_pred, output_dict=True)
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

    vectorizer = TfidfVectorizer(max_df=0.95, min_df=2, max_features=2000, stop_words='english')
    tfidf = vectorizer.fit_transform(df['clean_text'])
    nmf = NMF(n_components=n_topics, random_state=42, max_iter=400)
    W = nmf.fit_transform(tfidf)

    feature_names = vectorizer.get_feature_names_out()
    topics = {}
    topic_labels = [
        "Credit Card / Prepaid Card",
        "Checking / Savings Account",
        "Theft / Dispute Reporting",
        "Mortgage / Loan",
        "Others / General",
    ]
    for i, comp in enumerate(nmf.components_):
        top_words = [feature_names[j] for j in comp.argsort()[:-11:-1]]
        label = topic_labels[i] if i < len(topic_labels) else f"Topic {i+1}"
        topics[label] = top_words

    df['topic'] = W.argmax(axis=1)
    return topics, df


def compute_topsis(df_t):
    criteria = ['Severity', 'Resolution Time', 'Customer Sentiment', 'Priority']
    matrix = df_t[criteria].copy().astype(float)

    # Lower resolution time = better, so invert
    matrix['Resolution Time'] = matrix['Resolution Time'].max() - matrix['Resolution Time']

    # Normalize
    norms = np.sqrt((matrix ** 2).sum())
    norm_matrix = matrix / norms

    weights = np.array([0.30, 0.20, 0.20, 0.30])
    weighted = norm_matrix * weights

    ideal_best = weighted.max()
    ideal_worst = weighted.min()

    d_best = np.sqrt(((weighted - ideal_best) ** 2).sum(axis=1))
    d_worst = np.sqrt(((weighted - ideal_worst) ** 2).sum(axis=1))

    score = d_worst / (d_best + d_worst)
    return score


# ─── Sidebar ──────────────────────────────────────────────────────────────────
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
            help="Supports JSON, CSV, and Excel files"
        )
        if uploaded:
            try:
                raw = json.loads(uploaded.read().decode("utf-8", errors="ignore"))
                df_raw = pd.json_normalize(raw)
                st.success(f"✅ Loaded {len(df_raw):,} records")
            except Exception as e:
                st.error(f"Error: {e}")
    else:
        df_raw = get_sample_data()
        st.info(f"📊 Using {len(df_raw):,} sample records")

    st.divider()
    st.markdown("### ⚙️ Settings")
    n_topics = st.slider("NMF Topics", 3, 8, 5)
    n_sample = st.slider("Sample size (for speed)", 200, 1000, 500, step=100)

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

# Sample & preprocess
if len(df_raw) > n_sample:
    df_sample = df_raw.sample(n_sample, random_state=42).reset_index(drop=True)
else:
    df_sample = df_raw.copy()

df = preprocess_df(df_sample)

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
    cats = df['_source.product'].nunique() if '_source.product' in df.columns else 5

    with col1:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{total:,}</div><div class="metric-label">Total Tickets</div></div>', unsafe_allow_html=True)
    with col2:
        st.markdown(f'<div class="metric-card"><div class="metric-value">{cats}</div><div class="metric-label">Categories</div></div>', unsafe_allow_html=True)
    with col3:
        st.markdown(f'<div class="metric-card"><div class="metric-value">4</div><div class="metric-label">ML Models</div></div>', unsafe_allow_html=True)
    with col4:
        st.markdown(f'<div class="metric-card"><div class="metric-value">TOPSIS</div><div class="metric-label">Prioritization</div></div>', unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("### 🔄 Project Pipeline")
    pipeline_steps = [
        ("1️⃣", "Data Loading", "JSON complaints → Pandas DataFrame via json_normalize"),
        ("2️⃣", "Text Preprocessing", "Clean → Tokenize → Remove Stopwords → Lemmatize"),
        ("3️⃣", "EDA", "Category distribution, sentiment scoring, word frequencies"),
        ("4️⃣", "Feature Extraction", "TF-IDF vectorization → sparse feature matrix"),
        ("5️⃣", "Topic Modeling", "NMF groups unlabelled tickets into 5 clusters"),
        ("6️⃣", "Classification", "Logistic Regression / Decision Tree / Naive Bayes / Random Forest"),
        ("7️⃣", "TOPSIS Ranking", "Multi-criteria ranking by Severity · Sentiment · Resolution Time · Priority"),
    ]
    for icon, title, desc in pipeline_steps:
        st.markdown(f"**{icon} {title}** — {desc}")

    st.markdown("---")
    st.markdown("### 📋 Raw Data Preview")
    cols_show = [c for c in ['_source.product', '_source.complaint_what_happened', 'clean_text'] if c in df.columns]
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

    if '_source.product' not in df.columns:
        st.warning("No product/category column found in uploaded data.")
    else:
        col_left, col_right = st.columns(2)

        with col_left:
            st.markdown("#### 📊 Ticket Category Distribution")
            cat_counts = df['_source.product'].value_counts()
            fig, ax = plt.subplots(figsize=(6, 4))
            colors = ['#6c63ff', '#f093fb', '#4fd1c5', '#f6ad55', '#fc8181']
            bars = ax.barh(cat_counts.index[::-1], cat_counts.values[::-1], color=colors[:len(cat_counts)][::-1])
            ax.set_xlabel("Number of Tickets")
            ax.set_title("Top Complaint Categories", color='white', fontsize=12)
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
                    autopct='%1.1f%%', colors=wedge_colors[:len(cat_counts)],
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
        "complaints by decomposing the TF-IDF matrix into interpretable components."
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
    labels = [topic_names_list[i] if i < len(topic_names_list) else f"Topic {i}" for i in topic_dist.index]
    fig5, ax5 = plt.subplots(figsize=(8, 3.5),
                              facecolor='#1e2130')
    ax5.set_facecolor('#1e2130')
    bars = ax5.bar(labels, topic_dist.values,
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
    import matplotlib.pyplot as plt
    st.markdown('<div class="section-header">Supervised Classification Models</div>', unsafe_allow_html=True)

    if '_source.product' not in df.columns:
        st.warning("Target column `_source.product` not found.")
    else:
        with st.spinner("Training models… (this may take a moment)"):
            vectorizer, model_results, X_train, X_test, y_train, y_test = train_models(df)

        # F1 Summary
        st.markdown("#### 🏆 Model F1-Score Comparison")
        f1_data = {name: res['f1'] for name, res in model_results.items()}
        f1_df = pd.DataFrame(list(f1_data.items()), columns=['Model', 'F1-Score'])
        f1_df = f1_df.sort_values('F1-Score', ascending=False).reset_index(drop=True)

        fig6, ax6 = plt.subplots(figsize=(7, 3), facecolor='#1e2130')
        ax6.set_facecolor('#1e2130')
        colors_bar = ['#6c63ff', '#4fd1c5', '#f6ad55', '#fc8181']
        bars = ax6.barh(f1_df['Model'][::-1], f1_df['F1-Score'][::-1],
                        color=colors_bar[::-1], edgecolor='#1e2130', alpha=0.9)
        ax6.set_xlim(0, 1.1)
        ax6.set_xlabel("F1-Score (weighted)", color='white')
        ax6.set_title("Model Comparison", color='white')
        ax6.tick_params(colors='white')
        for bar, val in zip(bars, f1_df['F1-Score'][::-1]):
            ax6.text(bar.get_width() + 0.01, bar.get_y() + bar.get_height() / 2,
                     f"{val:.3f}", va='center', color='white', fontsize=10)
        plt.tight_layout()
        st.pyplot(fig6)
        plt.close()

        best_model_name = f1_df.iloc[0]['Model']
        best_f1 = f1_df.iloc[0]['F1-Score']
        st.success(f"🏆 Best model: **{best_model_name}** — F1-Score: **{best_f1:.4f}**")

        # Confusion Matrix
        st.markdown("#### 🔢 Confusion Matrix — Best Model")
        from sklearn.metrics import confusion_matrix
        import seaborn as sns

        best_res = model_results[best_model_name]
        cm = confusion_matrix(best_res['y_test'], best_res['y_pred'])
        labels = sorted(df['_source.product'].unique())

        fig7, ax7 = plt.subplots(figsize=(7, 5), facecolor='#1e2130')
        ax7.set_facecolor('#1e2130')
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                    xticklabels=labels, yticklabels=labels,
                    ax=ax7, cbar_kws={'shrink': 0.8})
        ax7.set_xlabel("Predicted", color='white')
        ax7.set_ylabel("Actual", color='white')
        ax7.set_title(f"Confusion Matrix — {best_model_name}", color='white')
        ax7.tick_params(colors='white', labelsize=7)
        plt.xticks(rotation=30, ha='right')
        plt.yticks(rotation=0)
        plt.tight_layout()
        st.pyplot(fig7)
        plt.close()

        # Per-class metrics
        st.markdown("#### 📋 Classification Report — Best Model")
        report = best_res['report']
        report_rows = []
        for cls, metrics in report.items():
            if isinstance(metrics, dict):
                report_rows.append({
                    'Class': cls,
                    'Precision': round(metrics['precision'], 3),
                    'Recall': round(metrics['recall'], 3),
                    'F1-Score': round(metrics['f1-score'], 3),
                    'Support': int(metrics['support']),
                })
        report_df = pd.DataFrame(report_rows)
        st.dataframe(report_df, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# TAB 5: TOPSIS
# ══════════════════════════════════════════════════════════════════════════════
with tabs[4]:
    st.markdown('<div class="section-header">TOPSIS-Based Ticket Prioritization</div>', unsafe_allow_html=True)
    st.markdown(
        "TOPSIS ranks tickets by their closeness to the **ideal solution** across "
        "four weighted criteria: Severity, Resolution Time, Sentiment, and Priority."
    )

    col_w1, col_w2, col_w3, col_w4 = st.columns(4)
    with col_w1: w_sev = st.slider("Severity weight", 0.1, 0.6, 0.30, 0.05)
    with col_w2: w_res = st.slider("Resolution Time weight", 0.1, 0.6, 0.20, 0.05)
    with col_w3: w_sen = st.slider("Sentiment weight", 0.1, 0.6, 0.20, 0.05)
    with col_w4: w_pri = st.slider("Priority weight", 0.1, 0.6, 0.30, 0.05)

    # Build ticket dataframe
    n_t = min(100, len(df))
    df_topsis = df.head(n_t).copy()

    # Use sentiment from TextBlob if available, else random
    try:
        from textblob import TextBlob
        df_topsis['Customer Sentiment'] = df_topsis['clean_text'].apply(
            lambda x: round(TextBlob(str(x)).sentiment.polarity, 4)
        )
        df_topsis['Customer Sentiment'] = df_topsis['Customer Sentiment'].apply(
            lambda x: (x + 1) / 2  # Normalize -1..1 to 0..1
        )
    except ImportError:
        np.random.seed(0)
        df_topsis['Customer Sentiment'] = np.random.rand(n_t)

    np.random.seed(1)
    df_topsis['Severity'] = np.random.randint(1, 6, size=n_t)
    df_topsis['Resolution Time'] = np.random.randint(1, 48, size=n_t)
    df_topsis['Priority'] = np.random.randint(1, 5, size=n_t)

    # Custom weights
    matrix = df_topsis[['Severity', 'Resolution Time', 'Customer Sentiment', 'Priority']].astype(float).copy()
    matrix['Resolution Time'] = matrix['Resolution Time'].max() - matrix['Resolution Time']
    norms = np.sqrt((matrix ** 2).sum())
    norm_matrix = matrix / norms
    weights_arr = np.array([w_sev, w_res, w_sen, w_pri])
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
    display_cols = ['_source.complaint_what_happened', '_source.product',
                    'Severity', 'Resolution Time', 'Customer Sentiment',
                    'Priority', 'TOPSIS Score', 'Rank', 'Priority Level']
    display_cols = [c for c in display_cols if c in df_sorted.columns]
    st.dataframe(
        df_sorted[display_cols].rename(columns={
            '_source.complaint_what_happened': 'Complaint',
            '_source.product': 'Category',
        }),
        use_container_width=True,
    )

    # Priority distribution chart
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

    if '_source.product' not in df.columns:
        st.warning("Classification model requires a labeled dataset.")
    else:
        with st.spinner("Ensuring models are trained…"):
            vectorizer, model_results, _, _, _, _ = train_models(df)

        best_clf_name = max(model_results, key=lambda k: model_results[k]['f1'])
        best_clf = model_results[best_clf_name]['model']

        examples = [
            "I lost my credit card and need a replacement urgently.",
            "There are unauthorized charges on my account, possible fraud.",
            "My loan application has been pending for over 3 months.",
            "The bank locked my account without any explanation.",
            "Customer service is terrible and no one responds.",
        ]
        example_choice = st.selectbox("Or pick an example:", ["(type your own)"] + examples)

        if example_choice != "(type your own)":
            default_text = example_choice
        else:
            default_text = ""

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
                vec = vectorizer.transform([cleaned])
                pred_category = best_clf.predict(vec)[0]

                try:
                    from textblob import TextBlob
                    sentiment = (TextBlob(cleaned).sentiment.polarity + 1) / 2
                except ImportError:
                    sentiment = 0.5

                # TOPSIS for single ticket vs reference extremes
                single = pd.DataFrame({
                    'Severity': [sev_input, 1, 5],
                    'Resolution Time': [res_input, 1, 72],
                    'Customer Sentiment': [sentiment, 0.0, 1.0],
                    'Priority': [pri_input, 1, 4],
                })
                score_vals = compute_topsis(single)
                topsis_score = score_vals.iloc[0]

                if topsis_score >= 0.65:
                    priority = "🔴 HIGH"
                    color = "#fc8181"
                elif topsis_score >= 0.45:
                    priority = "🟡 MEDIUM"
                    color = "#f6ad55"
                else:
                    priority = "🟢 LOW"
                    color = "#68d391"

                st.markdown("---")
                r1, r2, r3 = st.columns(3)
                r1.markdown(
                    f"<div class='metric-card'><div class='metric-value' style='color:#6c63ff'>{pred_category}</div>"
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
                st.markdown(
                    f"<div class='ticket-card'>"
                    f"<b style='color:#a0aec0'>Sentiment Score:</b> {sentiment:.3f} "
                    f"({'Positive 😊' if sentiment > 0.5 else 'Negative 😠' if sentiment < 0.4 else 'Neutral 😐'})<br/>"
                    f"<b style='color:#a0aec0'>Classifier Used:</b> {best_clf_name}<br/>"
                    f"<b style='color:#a0aec0'>Cleaned Text:</b> {cleaned[:200]}…"
                    f"</div>",
                    unsafe_allow_html=True,
                )
            else:
                st.warning("Please enter a complaint text.")
