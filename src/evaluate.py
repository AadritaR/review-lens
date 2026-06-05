import os
import sys
import pickle
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, f1_score,
    classification_report, confusion_matrix
)

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import LABEL_NAMES

RESULTS_DIR = "results"
MODELS_DIR  = "models"
LABEL_LIST  = [LABEL_NAMES[i] for i in sorted(LABEL_NAMES)]


# Confusion Matrix

def plot_confusion_matrix(y_true, y_pred, title, filename):
    os.makedirs(RESULTS_DIR, exist_ok=True)
    cm      = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    fig.suptitle(title, fontsize=13, fontweight='bold')

    for ax, data, fmt, subtitle in zip(
        axes,
        [cm, cm_norm],
        ['d', '.2%'],
        ['Counts', 'Normalized']
    ):
        sns.heatmap(data, annot=True, fmt=fmt, cmap='Blues',
                    xticklabels=LABEL_LIST, yticklabels=LABEL_LIST,
                    ax=ax, linewidths=0.5)
        ax.set_title(subtitle)
        ax.set_ylabel('True Label')
        ax.set_xlabel('Predicted Label')

    plt.tight_layout()
    out = os.path.join(RESULTS_DIR, filename)
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# Comparison Chart

def plot_comparison(baseline_metrics, distilbert_metrics):
    os.makedirs(RESULTS_DIR, exist_ok=True)

    categories = ['Accuracy', 'Weighted F1', 'Negative F1', 'Positive F1']

    def extract(m):
        return [
            m['accuracy'],
            m['weighted_f1'],
            m['per_class_f1']['Negative'],
            m['per_class_f1']['Positive'],
        ]

    b_vals = extract(baseline_metrics)
    d_vals = extract(distilbert_metrics)
    x = np.arange(len(categories))
    w = 0.35

    fig, ax = plt.subplots(figsize=(10, 5))
    bars1 = ax.bar(x - w/2, b_vals, w, label='TF-IDF + LR',
                   color='#5dade2', edgecolor='white')
    bars2 = ax.bar(x + w/2, d_vals, w, label='DistilBERT',
                   color='#2ecc71', edgecolor='white')

    for bar in bars1:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)
    for bar in bars2:
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.005,
                f'{bar.get_height():.3f}', ha='center', va='bottom', fontsize=9)

    for i, (b, d) in enumerate(zip(b_vals, d_vals)):
        ax.annotate(f'+{d-b:.3f}',
                    xy=(x[i] + w/2, d + 0.02),
                    ha='center', fontsize=8,
                    color='#27ae60', fontweight='bold')

    ax.set_title('TF-IDF + LR vs DistilBERT', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.set_ylabel('Score')
    ax.set_ylim(0.5, 1.05)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    plt.tight_layout()

    out = os.path.join(RESULTS_DIR, "metrics_comparison.png")
    plt.savefig(out, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved: {out}")


# Evaluate Baseline

def evaluate_baseline():
    from src.baseline_model import BaselineSentimentModel

    print("\n" + "="*50)
    print("EVALUATING: Baseline (TF-IDF + LR)")
    print("="*50)

    model = BaselineSentimentModel.load()

    with open(os.path.join(MODELS_DIR, "data_splits.pkl"), 'rb') as f:
        splits = pickle.load(f)

    X_test = splits['X_test']
    y_test = splits['y_test']

    y_pred = model.predict(X_test)
    acc    = accuracy_score(y_test, y_pred)
    f1_w   = f1_score(y_test, y_pred, average='weighted')
    f1_per = f1_score(y_test, y_pred, average=None, labels=[0, 1])

    print(f"Accuracy    : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Weighted F1 : {f1_w:.4f}")
    print(f"\n{classification_report(y_test, y_pred, target_names=LABEL_LIST)}")

    plot_confusion_matrix(y_test, y_pred,
                          "Baseline (TF-IDF + LR)",
                          "confusion_matrix_baseline.png")

    return {
        'accuracy':    acc,
        'weighted_f1': f1_w,
        'per_class_f1': {
            'Negative': float(f1_per[0]),
            'Positive': float(f1_per[1]),
        }
    }


# Evaluate DistilBERT 

def evaluate_distilbert():
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from torch.utils.data import DataLoader
    from src.dataset import ReviewDataset

    print("\n" + "="*50)
    print("EVALUATING: DistilBERT")
    print("="*50)

    device    = torch.device('cpu')
    model_dir = os.path.join(MODELS_DIR, "distilbert_sentiment")

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
    model.to(device)
    model.eval()

    with open(os.path.join(MODELS_DIR, "distilbert_test_data.pkl"), 'rb') as f:
        splits = pickle.load(f)

    X_test = splits['X_test']
    y_test = splits['y_test']

    dataset = ReviewDataset(X_test, y_test, tokenizer)
    loader  = DataLoader(dataset, batch_size=32, shuffle=False)

    all_preds = []
    print(f"Running inference on {len(X_test):,} test samples...")

    with torch.no_grad():
        for i, batch in enumerate(loader):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            outputs        = model(input_ids=input_ids, attention_mask=attention_mask)
            preds          = torch.argmax(outputs.logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            if (i + 1) % 10 == 0:
                print(f"  {i+1}/{len(loader)} batches done...")

    acc    = accuracy_score(y_test, all_preds)
    f1_w   = f1_score(y_test, all_preds, average='weighted')
    f1_per = f1_score(y_test, all_preds, average=None, labels=[0, 1])

    print(f"Accuracy    : {acc:.4f} ({acc*100:.2f}%)")
    print(f"Weighted F1 : {f1_w:.4f}")
    print(f"\n{classification_report(y_test, all_preds, target_names=LABEL_LIST)}")

    plot_confusion_matrix(y_test, all_preds,
                          "DistilBERT (Fine-tuned)",
                          "confusion_matrix_distilbert.png")

    return {
        'accuracy':    acc,
        'weighted_f1': f1_w,
        'per_class_f1': {
            'Negative': float(f1_per[0]),
            'Positive': float(f1_per[1]),
        }
    }


# Main

if __name__ == "__main__":
    b_metrics = evaluate_baseline()
    d_metrics = evaluate_distilbert()
    plot_comparison(b_metrics, d_metrics)

    print("\n" + "="*50)
    print("COMPARISON SUMMARY")
    print("="*50)
    print(f"{'Metric':<20} {'Baseline':>10} {'DistilBERT':>12} {'Delta':>8}")
    print("-" * 52)
    for key, label in [('accuracy', 'Accuracy'),
                        ('weighted_f1', 'Weighted F1')]:
        b = b_metrics[key]
        d = d_metrics[key]
        print(f"{label:<20} {b:>10.4f} {d:>12.4f} {d-b:>+8.4f}")
    for cls in ['Negative', 'Positive']:
        b = b_metrics['per_class_f1'][cls]
        d = d_metrics['per_class_f1'][cls]
        print(f"F1 {cls:<17} {b:>10.4f} {d:>12.4f} {d-b:>+8.4f}")