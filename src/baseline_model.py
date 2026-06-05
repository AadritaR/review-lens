"""
baseline_model.py
Stage 1
2-class: Negative (0) / Positive (1)
"""

import os
import sys
import pickle
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import load_and_prepare_dataset, preprocess_dataframe, LABEL_NAMES

MODELS_DIR  = "models"
RESULTS_DIR = "results"


class BaselineSentimentModel:

    def __init__(self):
        self.vectorizer = TfidfVectorizer(
            max_features=10000,
            ngram_range=(1, 2),
            sublinear_tf=True,
            min_df=2,
        )
        self.classifier = LogisticRegression(
            C=1.0,
            max_iter=1000,
            class_weight='balanced',
            solver='lbfgs',
            random_state=42,
        )

    def fit(self, texts, labels):
        print("Fitting TF-IDF vectorizer...")
        X = self.vectorizer.fit_transform(texts)
        print(f"  Matrix shape: {X.shape}")
        print("Training Logistic Regression...")
        self.classifier.fit(X, labels)
        print("  Done.")
        return self

    def predict(self, texts):
        X = self.vectorizer.transform(texts)
        return self.classifier.predict(X)

    def predict_proba(self, texts):
        X = self.vectorizer.transform(texts)
        return self.classifier.predict_proba(X)

    def predict_single(self, text):
        proba = self.predict_proba([text])[0]
        label = int(np.argmax(proba))
        return {
            'label':      label,
            'label_name': LABEL_NAMES[label],
            'confidence': float(proba[label]),
            'probabilities': {LABEL_NAMES[i]: float(p) for i, p in enumerate(proba)},
        }

    def save(self):
        os.makedirs(MODELS_DIR, exist_ok=True)
        with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), 'wb') as f:
            pickle.dump(self.vectorizer, f)
        with open(os.path.join(MODELS_DIR, "logistic_regression.pkl"), 'wb') as f:
            pickle.dump(self.classifier, f)
        print("Baseline model saved.")

    @classmethod
    def load(cls):
        model = cls.__new__(cls)
        with open(os.path.join(MODELS_DIR, "tfidf_vectorizer.pkl"), 'rb') as f:
            model.vectorizer = pickle.load(f)
        with open(os.path.join(MODELS_DIR, "logistic_regression.pkl"), 'rb') as f:
            model.classifier = pickle.load(f)
        return model


def evaluate(model, texts, labels, name="Model"):
    print(f"\n{'='*50}")
    print(f"RESULTS: {name}")
    print(f"{'='*50}")
    y_pred = model.predict(texts)
    acc  = accuracy_score(labels, y_pred)
    f1   = f1_score(labels, y_pred, average='weighted')
    print(f"Accuracy    : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Weighted F1 : {f1:.4f}")
    print(f"\n{classification_report(labels, y_pred, target_names=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)])}")

    # Confusion matrix
    cm = confusion_matrix(labels, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)],
                yticklabels=[LABEL_NAMES[i] for i in sorted(LABEL_NAMES)], ax=ax)
    ax.set_title(f"{name} — Confusion Matrix")
    ax.set_ylabel("True Label")
    ax.set_xlabel("Predicted Label")
    plt.tight_layout()
    os.makedirs(RESULTS_DIR, exist_ok=True)
    out = os.path.join(RESULTS_DIR, "confusion_matrix_baseline.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Confusion matrix saved → {out}")
    return {'accuracy': acc, 'weighted_f1': f1}


if __name__ == "__main__":
    import pandas as pd

    df = pd.read_csv("data/sample_reviews.csv")
    df = preprocess_dataframe(df, for_bert=False)

    X = df['clean_text'].tolist()
    y = df['label'].tolist()

    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=42
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.111, stratify=y_temp, random_state=42
    )

    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    model = BaselineSentimentModel()
    model.fit(X_train, y_train)

    evaluate(model, X_val,  y_val,  name="Baseline — Validation")
    metrics = evaluate(model, X_test, y_test, name="Baseline — Test")

    model.save()

    # Save test split
    with open(os.path.join(MODELS_DIR, "data_splits.pkl"), 'wb') as f:
        pickle.dump({'X_test': X_test, 'y_test': y_test}, f)
    print("\nTest split saved for DistilBERT comparison.")

    # Demo
    demos = [
        "Absolutely love this product! Works perfectly.",
        "Terrible quality. Broke after two days. Complete waste of money.",
        "Not bad but not great either. Shipping was slow.",
    ]
    print(f"\n{'='*50}")
    print("DEMO PREDICTIONS")
    print(f"{'='*50}")
    from src.preprocess import clean_text
    for d in demos:
        result = model.predict_single(clean_text(d, for_bert=False))
        print(f"\n  Review    : {d}")
        print(f"  Predicted : {result['label_name']} ({result['confidence']:.1%})")