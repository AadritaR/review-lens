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

```text
review-lens/
├── data/
│   └── sample_reviews.csv          ← 100k balanced subset
├── src/
│   ├── preprocess.py               ← text cleaning pipeline
│   ├── data_loader.py              ← sampling + label mapping
│   ├── baseline_model.py           ← TF-IDF + Logistic Regression
│   ├── dataset.py                  ← PyTorch Dataset class
│   ├── train.py                    ← DistilBERT fine-tuning
│   ├── evaluate.py                 ← metrics + confusion matrix
│   └── aspect_sentiment.py         ← aspect-based analysis
├── models/                         ← trained model weights
├── results/                        ← confusion matrices, comparison charts
├── app.py                          ← Streamlit app
└── requirements.txt
```
