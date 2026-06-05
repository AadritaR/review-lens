import os
import sys
import torch
import numpy as np
import streamlit as st

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))

st.set_page_config(
    page_title="Review Lens",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

st.markdown("""
<style>
    .stApp { background: #0f0a1e; }
    .block-container { padding-top: 2rem; max-width: 780px; }
    .app-title {
        font-size: 1.8rem; font-weight: 600;
        background: linear-gradient(135deg, #a78bfa, #7c3aed);
        -webkit-background-clip: text; -webkit-text-fill-color: transparent;
    }
    .app-sub { color: #6b7280; font-size: 0.9rem; margin-bottom: 1.5rem; }
    .section-label {
        font-size: 0.7rem; font-weight: 600; letter-spacing: 0.08em;
        text-transform: uppercase; color: #7c3aed; margin-bottom: 0.5rem;
    }
    .sentiment-box {
        text-align: center; padding: 1.2rem;
        border-radius: 12px; margin-bottom: 0.5rem;
    }
    .sentiment-pos { background: #052e16; border: 1px solid #166534; }
    .sentiment-neg { background: #1c0a0a; border: 1px solid #7f1d1d; }
    .sentiment-text-pos { font-size: 1.6rem; font-weight: 700; color: #4ade80; }
    .sentiment-text-neg { font-size: 1.6rem; font-weight: 700; color: #f87171; }
    .conf-text { font-size: 0.85rem; color: #9ca3af; margin-top: 2px; }
    .model-row {
        display: flex; justify-content: space-between; align-items: center;
        padding: 0.5rem 0; border-bottom: 1px solid #1e1033; font-size: 0.85rem;
    }
    .model-name { color: #9ca3af; }
    .pred-pos { color: #4ade80; font-weight: 600; }
    .pred-neg { color: #f87171; font-weight: 600; }
    .agree-badge {
        background: #052e16; border: 1px solid #166534;
        border-radius: 8px; padding: 6px 12px;
        font-size: 0.8rem; color: #4ade80; margin-top: 0.5rem;
    }
    .disagree-badge {
        background: #1c1208; border: 1px solid #92400e;
        border-radius: 8px; padding: 6px 12px;
        font-size: 0.8rem; color: #fb923c; margin-top: 0.5rem;
    }
    .aspect-row {
        display: grid; grid-template-columns: 26px 1fr 120px;
        align-items: center; gap: 10px;
        padding: 0.5rem 0; border-bottom: 1px solid #1e1033;
        font-size: 0.85rem;
    }
    .a-icon { font-size: 1rem; }
    .a-name { font-weight: 500; color: #e2e8f0; }
    .a-pos { color: #4ade80; text-align: right; }
    .a-neg { color: #f87171; text-align: right; }
    .a-none { color: #4b5563; text-align: right; }
    .metric-grid {
        display: grid; grid-template-columns: repeat(4,1fr);
        gap: 10px; margin-top: 0.5rem;
    }
    .metric-box {
        background: #1a0f35; border: 1px solid #2d1b69;
        border-radius: 10px; padding: 0.75rem 0.5rem; text-align: center;
    }
    .metric-val { font-size: 1.1rem; font-weight: 600; color: #e2e8f0; }
    .metric-lbl { font-size: 0.7rem; color: #9ca3af; margin-top: 2px; }
    .metric-delta { font-size: 0.7rem; color: #a78bfa; margin-top: 1px; }
    .stTextArea textarea {
    background-color: #ffffff !important;
    border: 1px solid #2d1b69 !important;
    border-radius: 10px !important;
    color: #000000 !important;
    -webkit-text-fill-color: #000000 !important;
}
.stTextArea textarea::placeholder {
    color: #666666 !important;
    -webkit-text-fill-color: #666666 !important;
}
.stSelectbox div[data-baseweb="select"] {
    background: #ffffff !important;
    border: 1px solid #2d1b69 !important;
}
.stSelectbox div[data-baseweb="select"] * {
    color: #111111 !important;
}
    .stButton button {
        background: linear-gradient(135deg, #7c3aed, #4f46e5) !important;
        color: white !important; border: none !important;
        border-radius: 10px !important; font-weight: 600 !important;
    }
    .stButton button:hover { opacity: 0.9 !important; }
    hr { border-color: #1e1033 !important; }
    p, label, .stMarkdown { color: #e2e8f0 !important; }
    .stExpander { background: #1a0f35 !important; border: 1px solid #2d1b69 !important; }
</style>
""", unsafe_allow_html=True)


# ── Model Loading ──────────────────────────────────────────────────────────────

@st.cache_resource(show_spinner="Loading baseline model...")
def load_baseline():
    try:
        from src.baseline_model import BaselineSentimentModel
        return BaselineSentimentModel.load()
    except:
        return None

@st.cache_resource(show_spinner="Loading DistilBERT...")
def load_analyzer():
    try:
        from src.aspect_sentiment import AspectAnalyzer
        return AspectAnalyzer()
    except:
        return None


# Helpers

ASPECT_ICONS = {
    'Quality': '🔧', 'Price': '💰',
    'Delivery': '📦', 'Customer Service': '🎧', 'Packaging': '📫',
}

def sentiment_box(label, confidence):
    css = "sentiment-pos" if label == "Positive" else "sentiment-neg"
    txt = "sentiment-text-pos" if label == "Positive" else "sentiment-text-neg"
    icon = "🟢" if label == "Positive" else "🔴"
    st.markdown(f"""
    <div class="sentiment-box {css}">
        <div class="{txt}">{icon} {label}</div>
        <div class="conf-text">{confidence:.1%} confidence</div>
    </div>""", unsafe_allow_html=True)

def aspect_row(aspect, sentiment):
    icon = ASPECT_ICONS.get(aspect, '•')
    if sentiment == 'Positive':
        cls, sym = 'a-pos', '✅ Positive'
    elif sentiment == 'Negative':
        cls, sym = 'a-neg', '❌ Negative'
    else:
        cls, sym = 'a-none', '— Not mentioned'
    st.markdown(f"""
    <div class="aspect-row">
        <span class="a-icon">{icon}</span>
        <span class="a-name">{aspect}</span>
        <span class="{cls}">{sym}</span>
    </div>""", unsafe_allow_html=True)


# Header

st.markdown('<div class="app-title">🔍 Review Lens</div>', unsafe_allow_html=True)
st.markdown('<div class="app-sub">Aspect-Based Sentiment Analysis · TF-IDF + LR vs DistilBERT</div>', unsafe_allow_html=True)

# Samples

SAMPLES = {
    "Select a sample...": "",
    "Mixed (Quality ✅ / Delivery ❌)":
        "The build quality is excellent but it arrived damaged and the price is too high for what you get.",
    "Strongly positive":
        "Absolutely love this! Great quality, arrived super fast, and customer service was fantastic.",
    "Strongly negative":
        "Terrible. Broke after 2 days. Customer service never responded. Complete waste of money.",
    "Ambiguous":
        "It's okay. Does what it says. Nothing special. Arrived on time. Pricing feels a bit high.",
    "Sarcastic":
        "Oh great, another product that stopped working after a week. Really impressive.",
    "Multi-aspect":
        "Shipping was fast and packaging was perfect. But the quality feels cheap. Customer service gave me a refund quickly.",
}

sample = st.selectbox("", list(SAMPLES.keys()), label_visibility="collapsed")
review_text = st.text_area(
    "Paste your Amazon review:",
    value=SAMPLES[sample],
    height=120,
    placeholder="Paste any Amazon review here...",
)

col_btn, col_opt = st.columns([3, 1])
with col_btn:
    analyze = st.button("🔍 Analyze review", type="primary", use_container_width=True)
with col_opt:
    show_sentences = st.toggle("Sentences", value=False)

if not analyze:
    st.markdown("---")
    st.markdown('<div class="section-label">Model performance · test set (100k reviews)</div>', unsafe_allow_html=True)
    st.markdown("""
    <div class="metric-grid">
        <div class="metric-box">
            <div class="metric-val">88.9%</div>
            <div class="metric-lbl">Baseline accuracy</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">90.7%</div>
            <div class="metric-lbl">DistilBERT accuracy</div>
            <div class="metric-delta">+1.77%</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">0.889</div>
            <div class="metric-lbl">Baseline F1</div>
        </div>
        <div class="metric-box">
            <div class="metric-val">0.907</div>
            <div class="metric-lbl">DistilBERT F1</div>
            <div class="metric-delta">+0.018</div>
        </div>
    </div>
    """, unsafe_allow_html=True)
    st.stop()

if not review_text.strip():
    st.warning("Please enter a review.")
    st.stop()

# Run Models

baseline_result   = None
distilbert_result = None
aspect_analysis   = None

with st.spinner("Analyzing..."):
    from src.preprocess import clean_text

    baseline_model = load_baseline()
    if baseline_model:
        cleaned = clean_text(review_text, for_bert=False)
        baseline_result = baseline_model.predict_single(cleaned)

    analyzer = load_analyzer()
    if analyzer:
        aspect_analysis   = analyzer.analyze(review_text)
        distilbert_result = aspect_analysis['overall']

st.markdown("---")

# Results

main = distilbert_result or baseline_result
if not main:
    st.error("No models loaded. Run baseline_model.py and train.py first.")
    st.stop()

# Overall + model comparison side by side
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="section-label">Overall sentiment</div>', unsafe_allow_html=True)
    sentiment_box(main['label_name'], main['confidence'])

with col2:
    st.markdown('<div class="section-label">Model comparison</div>', unsafe_allow_html=True)
    if baseline_result:
        cls = "pred-pos" if baseline_result['label_name'] == "Positive" else "pred-neg"
        st.markdown(f"""
        <div class="model-row">
            <span class="model-name">TF-IDF + LR</span>
            <span class="{cls}">{baseline_result['label_name']} · {baseline_result['confidence']:.0%}</span>
        </div>""", unsafe_allow_html=True)
    if distilbert_result:
        cls = "pred-pos" if distilbert_result['label_name'] == "Positive" else "pred-neg"
        st.markdown(f"""
        <div class="model-row">
            <span class="model-name">DistilBERT</span>
            <span class="{cls}">{distilbert_result['label_name']} · {distilbert_result['confidence']:.0%}</span>
        </div>""", unsafe_allow_html=True)

    if baseline_result and distilbert_result:
        if baseline_result['label_name'] == distilbert_result['label_name']:
            st.markdown('<div class="agree-badge">✅ Both models agree</div>', unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="disagree-badge">⚠️ Models disagree — trust DistilBERT</div>',
                unsafe_allow_html=True)

# Aspect Breakdown

if aspect_analysis:
    st.markdown("---")
    st.markdown('<div class="section-label">Aspect breakdown</div>', unsafe_allow_html=True)
    for asp, sent in aspect_analysis['aspects'].items():
        aspect_row(asp, sent)

    if show_sentences:
        st.markdown("---")
        st.markdown('<div class="section-label">Sentence breakdown</div>', unsafe_allow_html=True)
        for s in aspect_analysis['sentences']:
            if s['aspects']:
                with st.expander(s['sentence'][:80]):
                    st.write(f"**Aspects:** {', '.join(s['aspects'])}")
                    st.write(f"**Sentiment:** {s['sentiment']} ({s.get('confidence',0):.0%})")

# Footer metrics

st.markdown("---")
st.markdown('<div class="section-label">Model performance · test set (100k reviews)</div>', unsafe_allow_html=True)
st.markdown("""
<div class="metric-grid">
    <div class="metric-box">
        <div class="metric-val">88.9%</div>
        <div class="metric-lbl">Baseline accuracy</div>
    </div>
    <div class="metric-box">
        <div class="metric-val">90.7%</div>
        <div class="metric-lbl">DistilBERT accuracy</div>
        <div class="metric-delta">+1.77%</div>
    </div>
    <div class="metric-box">
        <div class="metric-val">0.889</div>
        <div class="metric-lbl">Baseline F1</div>
    </div>
    <div class="metric-box">
        <div class="metric-val">0.907</div>
        <div class="metric-lbl">DistilBERT F1</div>
        <div class="metric-delta">+0.018</div>
    </div>
</div>
""", unsafe_allow_html=True)
