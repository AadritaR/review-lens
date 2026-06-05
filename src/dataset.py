import torch
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer

MODEL_NAME = "distilbert-base-uncased"
MAX_LENGTH = 128


class ReviewDataset(Dataset):

    def __init__(self, texts, labels, tokenizer, max_length=MAX_LENGTH):
        self.texts     = texts
        self.labels    = labels
        self.tokenizer = tokenizer
        self.max_length = max_length
        assert len(texts) == len(labels)

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoding = self.tokenizer(
            str(self.texts[idx]),
            max_length=self.max_length,
            padding='max_length',
            truncation=True,
            return_attention_mask=True,
            return_tensors='pt',
        )
        return {
            'input_ids':      encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'labels':         torch.tensor(int(self.labels[idx]), dtype=torch.long),
        }


def load_tokenizer():
    print(f"Loading tokenizer: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    print(f"  Vocab size: {tokenizer.vocab_size:,}")
    return tokenizer


def create_dataloaders(X_train, y_train, X_val, y_val,
                       tokenizer, batch_size=16):
    train_ds = ReviewDataset(X_train, y_train, tokenizer)
    val_ds   = ReviewDataset(X_val,   y_val,   tokenizer)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=batch_size * 2, shuffle=False)

    print(f"  Train batches: {len(train_loader):,}")
    print(f"  Val batches  : {len(val_loader):,}")
    return train_loader, val_loader


if __name__ == "__main__":
    tokenizer = load_tokenizer()
    texts  = ["Great product!", "Terrible quality.", "Not bad at all."]
    labels = [1, 0, 1]
    ds     = ReviewDataset(texts, labels, tokenizer)
    sample = ds[0]
    print(f"\nSample input_ids shape : {sample['input_ids'].shape}")
    print(f"Sample label           : {sample['labels'].item()}")
    print(f"First 8 tokens         : {tokenizer.convert_ids_to_tokens(sample['input_ids'][:8].tolist())}")