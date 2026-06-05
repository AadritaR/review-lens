"""
aspect_sentiment.py
Stage 3: Aspect-Based Sentiment Analysis.

"""

import os
import sys
import torch
import numpy as np
from transformers import AutoModelForSequenceClassification, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.preprocess import clean_text, split_into_sentences, LABEL_NAMES

MODEL_DIR = "models/distilbert_sentiment"

# Aspect Keywords

ASPECT_KEYWORDS = {
    "Quality": [
        "quality", "material", "durable", "build", "sturdy",
        "finish", "construction", "well-made", "flimsy", "cheap",
        "solid", "premium", "feels", "looks", "workmanship",
    ],
    "Price": [
        "price", "expensive", "cheap", "affordable", "value",
        "cost", "worth", "overpriced", "budget", "deal",
        "money", "paid", "pay", "reasonable", "pricey",
    ],
    "Delivery": [
        "shipping", "delivery", "arrived", "arrive", "package",
        "courier", "fast", "slow", "delayed", "on time",
        "tracking", "dispatched", "received", "days", "weeks",
    ],
    "Customer Service": [
        "support", "customer service", "return", "refund",
        "response", "help", "staff", "representative",
        "contact", "complaint", "resolve", "helpful", "useless",
    ],
    "Packaging": [
        "packaging", "packed", "box", "wrapped", "wrapping",
        "damaged", "scratched", "dented", "broken", "intact",
        "bubble wrap", "protective",
    ],
}

NOT_MENTIONED = "Not mentioned"


# Predictor

class DistilBERTPredictor:
    

    def __init__(self, model_dir=MODEL_DIR):
        self.device    = torch.device('cpu')
        self.tokenizer = AutoTokenizer.from_pretrained(model_dir)
        self.model     = AutoModelForSequenceClassification.from_pretrained(model_dir)
        self.model.to(self.device)
        self.model.eval()

    def predict(self, text):
        encoding = self.tokenizer(
            text,
            max_length=128,
            padding='max_length',
            truncation=True,
            return_tensors='pt',
        )
        with torch.no_grad():
            logits = self.model(
                input_ids=encoding['input_ids'].to(self.device),
                attention_mask=encoding['attention_mask'].to(self.device)
            ).logits
            proba = torch.softmax(logits, dim=1).squeeze().cpu().numpy()

        label = int(np.argmax(proba))
        return {
            'label':         label,
            'label_name':    LABEL_NAMES[label],
            'confidence':    float(proba[label]),
            'probabilities': {LABEL_NAMES[i]: float(p) for i, p in enumerate(proba)},
        }


# Aspect Analyser

class AspectAnalyzer:

    def __init__(self, predictor=None):
        self.predictor = predictor or DistilBERTPredictor()

    def _detect_aspects(self, sentence):
        
        sentence_lower = sentence.lower()
        return [
            aspect for aspect, keywords in ASPECT_KEYWORDS.items()
            if any(kw in sentence_lower for kw in keywords)
        ]

    def _aggregate(self, results):
        
        if not results:
            return NOT_MENTIONED

        score = 0.0
        weight = 0.0
        for r in results:
            direction = 1 if r['label'] == 1 else -1
            score  += direction * r['confidence']
            weight += r['confidence']

        avg = score / weight if weight > 0 else 0
        if avg > 0.1:
            return LABEL_NAMES[1]   # Positive
        elif avg < -0.1:
            return LABEL_NAMES[0]   # Negative
        else:
            return "Neutral"

    def analyze(self, review_text):
        """
        Full aspect analysis for one review.
        Returns dict with overall sentiment and per-aspect breakdown.
        """
        # Overall sentiment on full review
        bert_text = clean_text(review_text, for_bert=True)
        overall   = self.predictor.predict(bert_text)

        # split
        sentences = split_into_sentences(review_text) or [review_text]

        
        from collections import defaultdict
        aspect_results  = defaultdict(list)
        sentence_detail = []

        for sentence in sentences:
            aspects = self._detect_aspects(sentence)
            if not aspects:
                sentence_detail.append({
                    'sentence': sentence,
                    'aspects':  [],
                    'sentiment': None,
                })
                continue

            result = self.predictor.predict(clean_text(sentence, for_bert=True))
            for aspect in aspects:
                aspect_results[aspect].append(result)

            sentence_detail.append({
                'sentence':   sentence,
                'aspects':    aspects,
                'sentiment':  result['label_name'],
                'confidence': result['confidence'],
            })

        
        aspects_out = {
            aspect: self._aggregate(aspect_results.get(aspect, []))
            for aspect in ASPECT_KEYWORDS
        }

        return {
            'overall':   overall,
            'aspects':   aspects_out,
            'sentences': sentence_detail,
        }

    def format(self, analysis, review_text=""):
        """Pretty print analysis result."""
        overall = analysis['overall']
        lines   = []

        if review_text:
            preview = review_text[:100] + ("..." if len(review_text) > 100 else "")
            lines.append(f'Review: "{preview}"')
            lines.append("")

        lines.append(f"Overall  : {overall['label_name']} "
                     f"(confidence: {overall['confidence']:.1%})")
        lines.append("")
        lines.append("Aspects:")
        lines.append("-" * 35)

        icons = {
            LABEL_NAMES[1]: "✅",
            LABEL_NAMES[0]: "❌",
            "Neutral":       "🟡",
            NOT_MENTIONED:   "—",
        }
        for aspect, sentiment in analysis['aspects'].items():
            lines.append(f"  {aspect:<20} {icons.get(sentiment,'—')}  {sentiment}")

        return "\n".join(lines)


# Demo

if __name__ == "__main__":
    analyzer = AspectAnalyzer()

    reviews = [
        "The build quality is excellent but it arrived damaged and the price is too high.",
        "Great product! Shipped really fast and packaging was perfect. Customer service helped me with a return quickly.",
        "Terrible quality. Broke after a week. Shipping was slow and support never responded.",
        "Decent product. Nothing special. Arrived on time. Price feels a bit high.",
    ]

    print("=" * 55)
    print("ASPECT SENTIMENT ANALYSIS — DEMO")
    print("=" * 55)

    for review in reviews:
        analysis = analyzer.analyze(review)
        print("\n" + "-" * 55)
        print(analyzer.format(analysis, review_text=review))
        print("\nSentence breakdown:")
        for s in analysis['sentences']:
            if s['aspects']:
                print(f"  · \"{s['sentence'][:70]}\"")
                print(f"    → {s['aspects']} : {s['sentiment']} "
                      f"({s.get('confidence', 0):.0%})")