import torch
import torch.nn as nn
from transformers import BertTokenizer, BertModel
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
import pandas as pd
from torch.optim import AdamW
import json
import random
from torch.cuda.amp import autocast, GradScaler

random.seed(42)

def load_pkl(path):
    df = pd.read_pickle(path)
    texts = df['content'].tolist()
    labels = df['label'].tolist()
    return texts, labels

def make_chunks_with_title(title_ids: torch.LongTensor,
                           content_ids: torch.LongTensor,
                           max_len: int,
                           stride: int,
                           pad_token_id: int):
    chunks = []
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id

    title_len = title_ids.size(0)
    max_content_chunk = max_len - title_len - 3
    if max_content_chunk <= 0:
        raise ValueError(f"标题太长（{title_len}），max_len={max_len} 不足以拼接标题。")

    seq_len = content_ids.size(0)
    start_positions = list(range(0, seq_len, stride))
    for start in start_positions:
        end = start + max_content_chunk
        if end <= seq_len:
            content_slice = content_ids[start:end]
            content_mask_slice = torch.ones_like(content_slice, dtype=torch.long)
        else:
            content_slice = content_ids[start:seq_len]
            content_mask_slice = torch.ones_like(content_slice, dtype=torch.long)
            pad_len = end - seq_len
            # 在末尾补 pad
            content_slice = torch.cat([
                content_slice,
                torch.full((pad_len,), pad_token_id, dtype=torch.long, device=content_ids.device)
            ], dim=0)
            content_mask_slice = torch.cat([
                content_mask_slice,
                torch.zeros(pad_len, dtype=torch.long, device=content_ids.device)
            ], dim=0)

        chunk_ids = torch.cat([
            torch.tensor([cls_id], dtype=torch.long, device=title_ids.device),
            title_ids,
            torch.tensor([sep_id], dtype=torch.long, device=title_ids.device),
            content_slice,
            torch.tensor([sep_id], dtype=torch.long, device=title_ids.device),
        ], dim=0)  # 长度 = 1 + title_len + 1 + max_content_chunk + 1 = max_len

        chunk_mask = torch.ones_like(chunk_ids, dtype=torch.long)

        chunks.append((chunk_ids, chunk_mask))
        if end >= seq_len:
            break

    return chunks


class LongTextDataset(Dataset):
    def __init__(self, entries: list, tokenizer: BertTokenizer, max_length: int = 512, stride: int = 256):
        self.entries = entries
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.stride = stride
        self.pad_token_id = tokenizer.pad_token_id

    def __len__(self):
        return len(self.entries)

    def __getitem__(self, idx):
        item = self.entries[idx]
        title = item["title"]
        content = item["content"]
        label = 1 if item.get("label") == "real" else 0
        category = item.get("category", None)

        title_enc = self.tokenizer(
            title,
            add_special_tokens=False,
            return_attention_mask=False,
            return_tensors="pt"
        )
        title_ids = title_enc["input_ids"].squeeze(0)  # Tensor([T])

        content_enc = self.tokenizer(
            content,
            add_special_tokens=False,
            return_attention_mask=False,
            return_tensors="pt"
        )
        content_ids = content_enc["input_ids"].squeeze(0)  # Tensor([C])

        chunks = make_chunks_with_title(
            title_ids=title_ids,
            content_ids=content_ids,
            max_len=self.max_length,
            stride=self.stride,
            pad_token_id=self.pad_token_id
        )

        return {
            "chunks": chunks,                # List of (Tensor([max_len]), Tensor([max_len]))
            "label": torch.tensor(label),
            "category": category
        }
    
def collate_fn(batch):
    input_ids_list = []
    attention_masks_list = []
    labels_list = []
    categories_list = []
    sample2chunk_cnt = []

    for sample in batch:
        chunks = sample["chunks"]       # List of (Tensor([max_len]), Tensor([max_len]))
        sample2chunk_cnt.append(len(chunks))
        for (chunk_ids, chunk_mask) in chunks:
            input_ids_list.append(chunk_ids)
            attention_masks_list.append(chunk_mask)
        labels_list.append(sample["label"])
        categories_list.append(sample["category"])

    input_ids_tensor = torch.stack(input_ids_list, dim=0)           # [total_chunks, max_len]
    attention_masks_tensor = torch.stack(attention_masks_list, dim=0)  # [total_chunks, max_len]
    labels_tensor = torch.stack(labels_list, dim=0)                 # [batch_size]

    return {
        "input_ids": input_ids_tensor,
        "attention_mask": attention_masks_tensor,
        "sample2chunk_cnt": sample2chunk_cnt,
        "labels": labels_tensor,
        "categories": categories_list
    }

class BERTClassifier(nn.Module):
    def __init__(self, pretrained_model_name: str, num_labels: int, dropout: float = 0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        hidden_size = self.bert.config.hidden_size  #  768

        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(hidden_size, num_labels)

    def forward(self, input_ids, attention_mask, sample2chunk_cnt):

        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        all_pooled = outputs.pooler_output  # [total_chunks, hidden_size]
        merged_logits = []
        chunk_ptr = 0
        for cnt in sample2chunk_cnt:
            this_chunk_pooled = all_pooled[chunk_ptr : chunk_ptr + cnt]  # [cnt, hidden_size]
            doc_rep = this_chunk_pooled.mean(dim=0)  # [hidden_size]
            doc_logit = self.classifier(self.dropout(doc_rep))  # [num_labels]
            merged_logits.append(doc_logit)
            chunk_ptr += cnt

        logits = torch.stack(merged_logits, dim=0)
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

            input_ids = batch["input_ids"].to(self.device)           
            attention_mask = batch["attention_mask"].to(self.device) 
            labels = batch["labels"].to(self.device)                  
            sample2chunk_cnt = batch["sample2chunk_cnt"]              

            with autocast():
                logits = self.model(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                    sample2chunk_cnt=sample2chunk_cnt
                )
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
        total_samples = 0
        with torch.no_grad():
            for batch in tqdm(data_loader, desc="Evaluating", leave=False):
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = batch["labels"].to(self.device)                 
                sample2chunk_cnt = batch["sample2chunk_cnt"]

                with autocast():
                    logits = self.model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        sample2chunk_cnt=sample2chunk_cnt
                    )  # [batch_size, num_labels]
                    loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)  # [batch_size]
                correct += (preds == labels).sum().item()
                total_samples += labels.size(0)

        avg_loss = total_loss / len(data_loader)
        accuracy = correct / total_samples
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
                    print(f"✅ Epoch {epoch}: New best valid_acc = {valid_acc:.4f}, model saved to {self.save_path}")

                print(f"Epoch {epoch}: train_loss={train_loss:.4f}, valid_loss={valid_loss:.4f}, valid_acc={valid_acc:.4f}")
            else:
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}")

        return history
    
if __name__ == "__main__":
    
    device = torch.device("cuda:4" if torch.cuda.is_available() else "cpu")

    with open("data/news_dataset/train.json", "r", encoding="utf-8") as f:
        all_data = json.load(f)
    with open("data/news_dataset/test.json", "r", encoding="utf-8") as f:
        test_data = json.load(f)
    random.shuffle(all_data)
    split_idx = int(len(all_data) * 0.8)
    train_samples = all_data[:split_idx]
    val_samples = all_data[split_idx:]
    tokenizer = BertTokenizer.from_pretrained("bert-base-chinese")
    max_len = 512
    stride = 384
    batch_size = 2
    train_dataset = LongTextDataset(
        entries    = train_samples,
        tokenizer  = tokenizer,
        max_length = max_len,
        stride     = stride
    )
    val_dataset = LongTextDataset(
        entries    = val_samples,
        tokenizer  = tokenizer,
        max_length = max_len,
        stride     = stride
    )
    test_dataset = LongTextDataset(
        entries    = test_data,
        tokenizer  = tokenizer,
        max_length = max_len,
        stride     = stride
    )
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_fn
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn
    )

    model = BERTClassifier("bert-base-chinese", num_labels=2, dropout=0.3)
    model.to(device)
    # optimizer = AdamW(model.parameters(), lr=2e-5)
    # criterion = nn.CrossEntropyLoss()
    # trainer = Trainer(model, optimizer, criterion, device, save_path="./best_model.pt")
    # trainer.train(epochs=3, train_loader=train_loader, valid_loader=val_loader)


    model.load_state_dict(torch.load("1.pt", map_location=device))
    model.eval()
    test_results = []
    with torch.no_grad():
        for batch in tqdm(test_loader, desc="Testing", leave=False):
            input_ids = batch["input_ids"].to(device)           # [total_chunks, 512]
            attention_mask = batch["attention_mask"].to(device) # [total_chunks, 512]
            labels = batch["labels"].to(device)                 # [batch_size]
            sample2chunk_cnt = batch["sample2chunk_cnt"]        # List[int]
            categories = batch["categories"]                     # List[str or None]

            logits = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                sample2chunk_cnt=sample2chunk_cnt
            )  # [batch_size, num_labels]
            preds = torch.argmax(logits, dim=1)  # [batch_size]

            for i in range(len(labels)):
                true_label = "real" if labels[i].item() == 1 else "fake"
                pred_label = "real" if preds[i].item() == 1 else "fake"
                test_results.append({
                    "category": categories[i] if categories[i] is not None else None,
                    "true_label": true_label,
                    "pred_label": pred_label
                })

    with open("baseline_results.json", "w", encoding="utf-8") as fout:
        json.dump(test_results, fout, ensure_ascii=False, indent=2)