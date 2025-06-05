import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import json
import random
from torch.cuda.amp import autocast, GradScaler
from torch.optim import AdamW

random.seed(42)

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
        raw_label = item.get("label", "").lower()
        label = 1 if raw_label == "real" else 0
        category = item.get("category", None)

        # 用 tokenizer.encode_plus 拼接 title + content，並截斷到 max_length
        encoding = self.tokenizer(
            text=title,
            text_pair=content,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation="only_second",   # 只在 content 部分截斷
            padding="max_length",       # 直接 pad 到 max_length
            return_attention_mask=True,
            return_tensors="pt"
        )
        input_ids = encoding["input_ids"].squeeze(0)         # [max_length]
        attention_mask = encoding["attention_mask"].squeeze(0)  # [max_length]

        return {
            "input_ids": input_ids,           # Tensor([512])
            "attention_mask": attention_mask, # Tensor([512])
            "label": torch.tensor(label, dtype=torch.long),
            "category": category
        }


def simple_collate_fn(batch):
    input_ids = torch.stack([item["input_ids"] for item in batch], dim=0)           # [B, 512]
    attention_masks = torch.stack([item["attention_mask"] for item in batch], dim=0)  # [B, 512]
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
        hidden_size = self.bert.config.hidden_size  # 通常 768

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask):
        # 直接傳給 BertModel，使用 pooler_output（[CLS] token 的隱藏向量）
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output  # [CLS]
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

        # 用於半精度的 GradScaler
        self.scaler = GradScaler()

    def train_epoch(self, data_loader):
        self.model.train()
        total_loss = 0.0

        for batch in tqdm(data_loader, desc="Training", leave=False):
            self.optimizer.zero_grad()

            input_ids = batch["input_ids"].to(self.device)           # [B,512]
            attention_mask = batch["attention_mask"].to(self.device) # [B,512]
            labels = batch["labels"].to(self.device)                 # [B]

            # 混合精度
            with autocast():
                logits = self.model(input_ids=input_ids, attention_mask=attention_mask)  # [B,2]
                loss = self.criterion(logits, labels)

            # scale & backward
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

                # 混合精度推理
                with autocast():
                    logits = self.model(input_ids=input_ids, attention_mask=attention_mask)
                    loss = self.criterion(logits, labels)

                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)  # [B]
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

    # 讀取 train.json、test.json
    with open("data/news_dataset/train.json", "r", encoding="utf-8") as f:
        all_data = json.load(f)
    with open("data/news_dataset/test.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)

    random.shuffle(all_data)
    split_idx = int(len(all_data) * 0.8)
    train_samples = all_data[:split_idx]
    val_samples = all_data[split_idx:]

    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")

    max_len = 256
    train_dataset = SimpleTextDataset(entries=train_samples,
                                      tokenizer=tokenizer,
                                      max_length=max_len)
    val_dataset = SimpleTextDataset(entries=val_samples,
                                    tokenizer=tokenizer,
                                    max_length=max_len)
    test_dataset = SimpleTextDataset(entries=test_data,
                                     tokenizer=tokenizer,
                                     max_length=max_len)

    # 建立 DataLoader
    batch_size = 8
    train_loader = DataLoader(train_dataset,
                              batch_size=batch_size,
                              shuffle=True,
                              collate_fn=simple_collate_fn)
    val_loader = DataLoader(val_dataset,
                            batch_size=batch_size,
                            shuffle=False,
                            collate_fn=simple_collate_fn)
    test_loader = DataLoader(test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             collate_fn=simple_collate_fn)

    # 定義模型、optimizer、loss
    num_labels = 2
    model = SimpleBERTClassifier(pretrained_model_name="bert-base-chinese",
                                 num_labels=num_labels,
                                 dropout=0.3)
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=2e-5)
    criterion = nn.CrossEntropyLoss()

    # 訓練
    trainer = Trainer(model, optimizer, criterion, device, save_path="./bert_model_our.pt")
    # trainer.train(epochs=1, train_loader=train_loader, valid_loader=val_loader)

    # 使用最佳模型做測試
    model.load_state_dict(torch.load("bert_model_our.pt", map_location=device))
    model.eval()
    test_results = []

    correct = 0
    total = 0

    from collections import defaultdict
    category_correct = defaultdict(int)
    category_total   = defaultdict(int)
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", leave=False):
            input_ids = batch["input_ids"].to(device)           # [B, max_len]
            attention_mask = batch["attention_mask"].to(device) # [B, max_len]
            labels = batch["labels"].to(device)                 # [B]
            categories = batch["categories"]  # [B]

            # 混合精度推理
            with autocast():
                logits = model(input_ids=input_ids, attention_mask=attention_mask)  # [B, 2]

            preds = torch.argmax(logits, dim=1)  # [B]

            correct += (preds == labels).sum().item()
            total   += labels.size(0)

            for i in range(labels.size(0)):
                cat = categories[i]  # 可能是字串或 None
                if cat is None:
                    continue

                category_total[cat] += 1
                if preds[i] == labels[i]:
                    category_correct[cat] += 1

    # 計算整體 accuracy
    accuracy = correct / total if total > 0 else 0.0
    print(f"Overall Test Accuracy: {accuracy:.4f} ({correct}/{total})\n")

    # 計算並列印每個 category 的 accuracy
    print("=== 各 category 測試準確度 ===")
    for cat, tot in category_total.items():
        corr = category_correct[cat]
        acc_cat = corr / tot if tot > 0 else 0.0
        print(f"Category: {cat:20s}  |  Accuracy: {acc_cat:.4f} ({corr}/{tot})")