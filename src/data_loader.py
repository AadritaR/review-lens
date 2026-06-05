import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.preprocess import load_and_prepare_dataset

if __name__ == "__main__":
    df = load_and_prepare_dataset(
        csv_path="data/reviews-raw.csv",
        n_samples=100000,
        random_state=42,
    )
    output_path = "data/sample_reviews.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df):,} reviews → {output_path}")
    print(df[['label', 'reviewText']].head(5).to_string())