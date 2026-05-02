import torch
import torch.nn as nn
from datasets import load_dataset
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_split
from torch.optim import AdamW
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from model import SpamClassifier


class SpamDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_len=128):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        encoded = self.tokenizer(
            self.texts[idx],
            truncation=True,
            padding="max_length",
            max_length=self.max_len,
            return_tensors="pt",
        )
        return {
            "input_ids": encoded["input_ids"].squeeze(0),
            "attention_mask": encoded["attention_mask"].squeeze(0),
            "labels": torch.tensor(self.labels[idx], dtype=torch.long),
        }


def load_training_data():
    dataset = load_dataset("sms_spam")
    train_split = dataset["train"]
    texts = [str(x) for x in train_split["sms"]]
    labels = [int(x) for x in train_split["label"]]
    return train_test_split(texts, labels, test_size=0.2, random_state=42, stratify=labels)


def main():
    train_texts, val_texts, train_labels, val_labels = load_training_data()

    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    train_dataset = SpamDataset(train_texts, train_labels, tokenizer)
    val_dataset = SpamDataset(val_texts, val_labels, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    model = SpamClassifier().to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5, weight_decay=0.01)
    criterion = nn.CrossEntropyLoss()
    epochs = 3

    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=len(train_loader),
        num_training_steps=epochs * len(train_loader),
    )

    for epoch in range(epochs):
        model.train()
        total_loss = 0.0

        for batch in train_loader:
            optimizer.zero_grad()

            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()

            total_loss += loss.item()

        model.eval()
        preds, true_labels = [], []
        with torch.no_grad():
            for batch in val_loader:
                input_ids = batch["input_ids"].to(device)
                attention_mask = batch["attention_mask"].to(device)
                labels = batch["labels"]

                logits = model(input_ids, attention_mask)
                preds.extend(logits.argmax(-1).cpu().numpy())
                true_labels.extend(labels.numpy())

        acc = accuracy_score(true_labels, preds)
        f1 = f1_score(true_labels, preds)
        print(f"Epoch {epoch + 1} | Loss: {total_loss:.3f} | Acc: {acc:.3f} | F1: {f1:.3f}")

    torch.save(model.state_dict(), "model.pt")
    print("Model saved as model.pt")


if __name__ == "__main__":
    main()