import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import random
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW
import pandas as pd

random.seed(42)
torch.manual_seed(42)

# 載入 .pkl 並轉換格式
def load_pkl_as_entries(path: str) -> list:
    df = pd.read_pickle(path)
    entries = []
    for _, row in df.iterrows():
        entries.append({
            "title": "",  # 這個範例沒有標題，統一留空
            "content": row["content"],
            "label": row["label"],  # 假設欄位值就是 "real" 或 "fake"
            "category": row.get("category", None)
        })
    return entries


class SimpleTextDataset(Dataset):
    def __init__(self, entries: list, tokenizer: BertTokenizer, max_length: int = 512):
        self.entries = entries
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]
        title = item.get("title", "")
        content = item.get("content", "")
        label = item.get("label", 0) 
        category = item.get("category", None)

        encoding = self.tokenizer(
            text=title,
            text_pair=content,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation="only_second",  # 只截 content 部分
            padding="max_length",
            return_attention_mask=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)         # [max_length]
        attention_mask = encoding["attention_mask"].squeeze(0)  # [max_length]

        return {
            "input_ids": input_ids,           # Tensor([max_length])
            "attention_mask": attention_mask, # Tensor([max_length])
            "label": torch.tensor(label, dtype=torch.long),
            "category": category
        }


def simple_collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)           # [B, max_length]
    attention_masks = torch.stack([item["attention_mask"] for item in batch], dim=0)  # [B, max_length]
    labels = torch.stack([item["label"] for item in batch], dim=0)                   # [B]
    categories = [item["category"] for item in batch]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "labels": labels,
        "categories": categories
    }


class SimpleBERTClassifier(nn.Module):
    def __init__(self, pretrained_model_name: str, num_labels: int, dropout: float = 0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        hidden_size = self.bert.config.hidden_size  # 通常是 768

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output  # [B, hidden_size]
        logits = self.classifier(self.dropout(pooled))  # [B, num_labels]
        return logits


class Trainer:
    def __init__(self, model, optimizer, criterion, device, save_path="./best_model.pt"):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.save_path = save_path
        self.best_acc = 0.0
        self.scaler = GradScaler()

    def train_epoch(self, data_loader):
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(data_loader, desc="Training", leave=False):
            self.optimizer.zero_grad()
            input_ids = batch["input_ids"].to(self.device)           # [B, max_length]
            attention_mask = batch["attention_mask"].to(self.device) # [B, max_length]
            labels = batch["labels"].to(self.device)                 # [B]

            with autocast():
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)  # [B, num_labels]
                loss = self.criterion(logits, labels)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()

        return total_loss / len(data_loader)

    def eval_epoch(self, data_loader):
        self.model.eval()
        total_loss = 0.0
        correct = 0
        total = 0

        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating", leave=False):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)

                with autocast():
                    logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = self.criterion(logits, labels)

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == labels).sum().item()
                total += labels.size(0)

        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total
        return avg_loss, accuracy

    def train(self, epochs, train_loader, valid_loader=None):
        history = {"train_loss": [], "valid_loss": [], "valid_acc": []}

        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            history["train_loss"].append(train_loss)

            if valid_loader is not None:
                valid_loss, valid_acc = self.eval_epoch(valid_loader)
                history["valid_loss"].append(valid_loss)
                history["valid_acc"].append(valid_acc)

                if valid_acc > self.best_acc:
                    self.best_acc = valid_acc
                    torch.save(self.model.state_dict(), self.save_path)
                    print(f"✅ Epoch {epoch}: New best valid_acc = {valid_acc:.4f}, saved to {self.save_path}")

                print(f"Epoch {epoch}: train_loss={train_loss:.4f}, valid_loss={valid_loss:.4f}, valid_acc={valid_acc:.4f}")
            else:
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}")

        return history


if __name__ == "__main__":
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

    # 1) 先把 train.pkl、val.pkl 載進來
    train_entries = load_pkl_as_entries("data/benchmark/weibo21/train.pkl")
    val_entries   = load_pkl_as_entries("data/benchmark/weibo21/val.pkl")
    # 如果還有 test.pkl，也可以同樣載入：
    # test_entries = load_pkl_as_entries("path/to/test.pkl")

    # 2) 初始化 tokenizer
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    # 3) 建 dataset 與 dataloader
    max_len = 256  # 你可以調整成 512 或 384，都取決於 GPU 記憶體
    batch_size = 8

    train_dataset = SimpleTextDataset(entries=train_entries,
                                      tokenizer=tokenizer,
                                      max_length=max_len)
    val_dataset = SimpleTextDataset(entries=val_entries,
                                    tokenizer=tokenizer,
                                    max_length=max_len)

    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=simple_collate_fn)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            collate_fn=simple_collate_fn)

    # 4) 建立模型、optimizer、loss
    num_labels = 2
    model = SimpleBERTClassifier(pretrained_model_name="bert-base-chinese",
                                 num_labels=num_labels,
                                 dropout=0.3)
    model.to(device)

    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    # 5) 開始訓練
    trainer = Trainer(model, optimizer, criterion, device, save_path="./bert_model_weibo.pt")
    trainer.train(epochs=10, train_loader=train_loader, valid_loader=val_loader)

    # 6) 如果你有測試集，可以在這裡載入 best model 並評估
    # model.load_state_dict(torch.load("./bert_chinese_model.pt", map_location=device))
    # model.eval()
    # test_entries = load_pkl_as_entries("path/to/test.pkl")
    # test_dataset = SimpleTextDataset(entries=test_entries, tokenizer=tokenizer, max_length=max_len)
    # test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=simple_collate_fn)
    # test_loss, test_acc = trainer.eval_epoch(test_loader)
    # print(f"Test loss: {test_loss:.4f}, Test acc: {test_acc:.4f}")

    print("訓練結束，模型已儲存在 ./bert_chinese_model.pt")