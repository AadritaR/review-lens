"""
preprocess.py
Used in Text cleaning pipeline for review sentiment analysis.
Dataset: Amazon Polarity (2-class: Negative / Positive)
Columns in CSV: no header, col0=label(1/2), col1=title, col2=review_text
"""

import re
import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize

# Download required NLTK data
def download_nltk_data():
    for resource in ['stopwords', 'wordnet', 'punkt', 'punkt_tab', 'omw-1.4']:
        nltk.download(resource, quiet=True)

download_nltk_data()

# Constants 

STOP_WORDS = set(stopwords.words('english'))
# Keep negation words — critical for sentiment
NEGATION_WORDS = {'no', 'not', "n't", 'never', 'neither', 'nothing', 'hardly', 'barely'}
STOP_WORDS -= NEGATION_WORDS

lemmatizer = WordNetLemmatizer()

LABEL_NAMES  = {0: 'Negative', 1: 'Positive'}
LABEL_COLORS = {0: '🔴', 1: '🟢'}


# Cleaning Functions 

def remove_html(text):
    return re.sub(r'<[^>]+>', ' ', text)

def remove_urls(text):
    return re.sub(r'http\S+|www\S+', '', text)

def expand_contractions(text):
    contractions = {
        "n't": " not", "'re": " are", "'s": " is", "'d": " would",
        "'ll": " will", "'ve": " have", "'m": " am",
        "won't": "will not", "can't": "cannot",
    }
    for k, v in contractions.items():
        text = text.replace(k, v)
    return text

def remove_punctuation(text):
    text = re.sub(r"[^\w\s']", ' ', text)
    text = re.sub(r"\s'\s|^'|'$", ' ', text)
    return text

def clean_text(text, for_bert=False):
    """
    Full cleaning pipeline.
    for_bert=False → removes stopwords + lemmatizes  (for TF-IDF)
    for_bert=True  → light cleaning only             (BERT handles the rest)
    """
    if not isinstance(text, str) or not text.strip():
        return ""

    text = text.lower()
    text = remove_html(text)
    text = remove_urls(text)
    text = expand_contractions(text)
    text = remove_punctuation(text)
    text = re.sub(r'\s+', ' ', text).strip()

    if for_bert:
        return text

    tokens = text.split()
    tokens = [t for t in tokens if t not in STOP_WORDS]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    tokens = [t for t in tokens if len(t) > 1]
    return ' '.join(tokens)


# Label Mapping

def map_label(raw_label):
    """
    Dataset uses 1=Negative, 2=Positive.
    We map to 0=Negative, 1=Positive.
    """
    return 0 if int(raw_label) == 1 else 1


# Dataset Loader

def load_and_prepare_dataset(csv_path, n_samples=50000, random_state=42):
    """
    Load raw CSV (no header), sample n_samples rows, return clean DataFrame.
    """
    print(f"Loading dataset from {csv_path} ...")
    df = pd.read_csv(
        csv_path,
        header=None,
        names=['raw_label', 'title', 'reviewText'],
        on_bad_lines='skip',
    )

    print(f"  Total rows loaded: {len(df):,}")
    df = df.dropna(subset=['reviewText', 'raw_label'])
    df = df[df['reviewText'].str.strip() != '']
    df['label'] = df['raw_label'].apply(map_label)

    # Stratified sample — equal Negative and Positive
    n_each = n_samples // 2
    df_neg = df[df['label'] == 0].sample(n=n_each, random_state=random_state)
    df_pos = df[df['label'] == 1].sample(n=n_each, random_state=random_state)
    df = pd.concat([df_neg, df_pos]).sample(frac=1, random_state=random_state).reset_index(drop=True)

    print(f"  Sampled: {len(df):,} reviews")
    print(f"  Negative: {(df['label']==0).sum():,} | Positive: {(df['label']==1).sum():,}")
    return df


def preprocess_dataframe(df, for_bert=False):
    """Add clean_text column to dataframe."""
    mode = "BERT" if for_bert else "TF-IDF"
    print(f"Cleaning text for {mode} ...")
    df = df.copy()
    df['clean_text'] = df['reviewText'].apply(lambda x: clean_text(x, for_bert=for_bert))
    df = df[df['clean_text'].str.strip() != ''].reset_index(drop=True)
    print(f"  Done. {len(df):,} reviews remaining.")
    return df


def split_into_sentences(text):
    """Split review into sentences (used by aspect analyser)."""
    sentences = sent_tokenize(text)
    return [s.strip() for s in sentences if len(s.split()) >= 3]


# Quick Test 

if __name__ == "__main__":
    tests = [
        "GREAT Product!!! Shipping took 3 weeks. Way too slow.",
        "<br/>Not bad, not great. Wouldn't buy again.",
        "Absolutely love it. Best purchase ever. Customer service was very helpful.",
        "Terrible. Broke after 2 days. Complete waste of money.",
    ]
    print("=" * 55)
    for t in tests:
        print(f"Original : {t}")
        print(f"TF-IDF   : {clean_text(t, for_bert=False)}")
        print(f"BERT     : {clean_text(t, for_bert=True)}")
        print()