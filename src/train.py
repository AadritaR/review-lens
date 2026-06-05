"""
train.py
Stage 2: Fine tuning
2-class: Negative (0) / Positive (1)
"""

import os
import sys
import json
import numpy as np
import torch
import torch.nn as nn
from torch.optim import AdamW
from transformers import AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import load_and_prepare_dataset, preprocess_dataframe, LABEL_NAMES
from src.dataset import load_tokenizer, create_dataloaders

MODEL_NAME  = "distilbert-base-uncased"
MAX_LENGTH  = 128
SAVE_DIR    = "models/distilbert_sentiment"
RESULTS_DIR = "results"


def train_epoch(model, loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss, total_correct, total = 0, 0, 0

    pbar = tqdm(loader, desc="  Training", ncols=80)
    for batch in pbar:
        input_ids      = batch['input_ids'].to(device)
        attention_mask = batch['attention_mask'].to(device)
        labels         = batch['labels'].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss    = criterion(outputs.logits, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        optimizer.step()
        scheduler.step()

        preds = torch.argmax(outputs.logits, dim=1)
        total_correct += (preds == labels).sum().item()
        total_loss    += loss.item() * len(labels)
        total         += len(labels)
        pbar.set_postfix({'loss': f"{total_loss/total:.4f}",
                          'acc':  f"{total_correct/total:.4f}"})

    return total_loss / total, total_correct / total


def eval_epoch(model, loader, criterion, device):
    model.eval()
    total_loss, total_correct, total = 0, 0, 0
    all_preds, all_labels = [], []

    with torch.no_grad():
        for batch in tqdm(loader, desc="  Evaluating", ncols=80):
            input_ids      = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels         = batch['labels'].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss    = criterion(outputs.logits, labels)
            preds   = torch.argmax(outputs.logits, dim=1)

            total_loss    += loss.item() * len(labels)
            total_correct += (preds == labels).sum().item()
            total         += len(labels)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    return (total_loss / total,
            total_correct / total,
            f1_score(all_labels, all_preds, average='weighted'))


def train(n_samples=10000, batch_size=16, num_epochs=3,
          learning_rate=2e-5, random_state=42):

    device = torch.device('cpu')
    print(f"Device: {device}")

    # 1. Load data
    import pandas as pd
    df = pd.read_csv("data/sample_reviews.csv")
    if n_samples < len(df):
        df_neg = df[df['label'] == 0].sample(n_samples // 2, random_state=random_state)
        df_pos = df[df['label'] == 1].sample(n_samples // 2, random_state=random_state)
        df = pd.concat([df_neg, df_pos]).sample(frac=1, random_state=random_state).reset_index(drop=True)
    df = preprocess_dataframe(df, for_bert=True)

    X = df['clean_text'].tolist()
    y = df['label'].tolist()
    print(f"Training on {len(X):,} reviews")

    # 2. Split
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=0.1, stratify=y, random_state=random_state
    )
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=0.111, stratify=y_temp, random_state=random_state
    )
    print(f"Train: {len(X_train):,} | Val: {len(X_val):,} | Test: {len(X_test):,}")

    # 3. Tokenizer + DataLoaders
    tokenizer = load_tokenizer()
    train_loader, val_loader = create_dataloaders(
        X_train, y_train, X_val, y_val, tokenizer, batch_size=batch_size
    )

    # 4. Model
    print(f"\nLoading model: {MODEL_NAME}")
    model = AutoModelForSequenceClassification.from_pretrained(
        MODEL_NAME, num_labels=2
    )
    model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    print(f"  Parameters: {total_params:,}")

    # 5. Optimizer + scheduler
    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=0.01)
    total_steps = len(train_loader) * num_epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=100, num_training_steps=total_steps
    )
    criterion = nn.CrossEntropyLoss()

    # 6. Training loop
    history = []
    best_val_f1 = 0.0
    os.makedirs(SAVE_DIR, exist_ok=True)

    for epoch in range(1, num_epochs + 1):
        print(f"\nEpoch {epoch}/{num_epochs}")
        train_loss, train_acc = train_epoch(
            model, train_loader, optimizer, scheduler, criterion, device
        )
        val_loss, val_acc, val_f1 = eval_epoch(
            model, val_loader, criterion, device
        )

        print(f"  Train — loss: {train_loss:.4f}  acc: {train_acc:.4f}")
        print(f"  Val   — loss: {val_loss:.4f}  acc: {val_acc:.4f}  f1: {val_f1:.4f}")

        history.append({
            'epoch': epoch,
            'train_loss': train_loss, 'train_acc': train_acc,
            'val_loss': val_loss, 'val_acc': val_acc, 'val_f1': val_f1
        })

        if val_f1 > best_val_f1:
            best_val_f1 = val_f1
            model.save_pretrained(SAVE_DIR)
            tokenizer.save_pretrained(SAVE_DIR)
            print(f"  ✓ Best model saved (val_f1={best_val_f1:.4f})")

    # 7. Save history + test split
    with open(os.path.join(SAVE_DIR, "history.json"), 'w') as f:
        json.dump(history, f, indent=2)

    import pickle
    with open("models/distilbert_test_data.pkl", 'wb') as f:
        pickle.dump({'X_test': X_test, 'y_test': y_test}, f)

    print(f"\nTraining complete. Best val F1: {best_val_f1:.4f}")
    return history


if __name__ == "__main__":
    train(n_samples=10000, batch_size=16, num_epochs=3)