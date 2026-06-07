# Review Lens 🔍

> Aspect-level sentiment analysis for product reviews · TF-IDF + Logistic Regression vs fine-tuned DistilBERT

---

## What it does

Star ratings tell you _how much_ a customer liked a product, but not _why_. A 3-star review is ambiguous — did the product fail on quality, arrive late, or feel overpriced?

**Review Lens** reads raw review text and extracts:

- **Overall sentiment** — Positive / Negative with confidence score
- **Aspect-level sentiment** — Per-dimension breakdown across 5 product aspects:
  - 🔧 Quality
  - 💰 Price
  - 📦 Delivery
  - 🎧 Customer Service
  - 📫 Packaging

---

## Results

| Model                        | Accuracy  | Weighted F1 |
| ---------------------------- | --------- | ----------- |
| TF-IDF + Logistic Regression | 88.9%     | 0.889       |
| **DistilBERT (fine-tuned)**  | **90.7%** | **0.907**   |

**+1.77% improvement** through contextual transformer representations over bag-of-words.

Tested on 10,000 held-out reviews from a balanced 100,000-review training set.

---

## Project Structure

review-lens/
├── data/
│ └── sample_reviews.csv
├── src/
│ ├── preprocess.py
│ ├── data_loader.py
│ ├── baseline_model.py
│ ├── dataset.py
│ ├── train.py
│ ├── evaluate.py
│ └── aspect_sentiment.py
├── models/
├── results/
├── app.py
└── requirements.txt

---
