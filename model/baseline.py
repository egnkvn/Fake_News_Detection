import os
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import BertTokenizer, BertModel, AdamW

# 1. Dataset class
class NewsDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        label = self.labels[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        item = {key: val.squeeze(0) for key, val in encoding.items()}
        item['labels'] = torch.tensor(label, dtype=torch.long)
        return item

# 2. LSTM baseline
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, output_dim, num_layers=1, dropout=0.3):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, num_layers=num_layers, bidirectional=True, batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_dim * 2, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask=None, **kwargs):
        # input_ids: [batch_size, seq_len]
        embedded = self.dropout(self.embedding(input_ids))  # [batch, seq, embed]
        outputs, (hidden, cell) = self.lstm(embedded)
        # concatenate final forward and backward hidden
        hidden = torch.cat((hidden[-2], hidden[-1]), dim=1)  # [batch, hidden*2]
        logits = self.fc(self.dropout(hidden))
        return logits

# 3. BERT baseline
class BERTClassifier(nn.Module):
    def __init__(self, pretrained_model_name, output_dim, dropout=0.3):
        super().__init__()
        self.bert = BertModel.from_pretrained(pretrained_model_name)
        hidden_size = self.bert.config.hidden_size
        self.fc = nn.Linear(hidden_size, output_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, input_ids, attention_mask, **kwargs):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        pooled = outputs.pooler_output  # [batch, hidden]
        return self.fc(self.dropout(pooled))

class Trainer:
    def __init__(self, model, dataloader, optimizer, criterion, device):
        self.model = model
        self.dataloader = dataloader
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.model.to(device)

    def train_epoch(self):
        self.model.train()
        total_loss = 0
        for batch in self.dataloader:
            self.optimizer.zero_grad()
            batch = {k: v.to(self.device) for k, v in batch.items()}
            logits = self.model(**batch)
            loss = self.criterion(logits, batch['labels'])
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(self.dataloader)

    def eval_epoch(self, dataloader):
        self.model.eval()
        total_loss, correct = 0, 0
        with torch.no_grad():
            for batch in dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                logits = self.model(**batch)
                loss = self.criterion(logits, batch['labels'])
                total_loss += loss.item()
                preds = torch.argmax(logits, dim=1)
                correct += (preds == batch['labels']).sum().item()
        avg_loss = total_loss / len(dataloader)
        accuracy = correct / len(dataloader.dataset)
        return avg_loss, accuracy

    def train(self, epochs, valid_dataloader=None):
        history = {'train_loss': [], 'valid_loss': [], 'valid_acc': []}
        for epoch in range(1, epochs + 1):
            train_loss = self.train_epoch()
            history['train_loss'].append(train_loss)

            if valid_dataloader:
                valid_loss, valid_acc = self.eval_epoch(valid_dataloader)
                history['valid_loss'].append(valid_loss)
                history['valid_acc'].append(valid_acc)
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}, "
                      f"valid_loss={valid_loss:.4f}, valid_acc={valid_acc:.4f}")
            else:
                print(f"Epoch {epoch}: train_loss={train_loss:.4f}")
        return history

# 5. Putting it together
if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Example data (replace with real dataset)
    texts = ["News claim example 1", "Another news claim"]
    labels = [0, 1]

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    max_len = 128
    dataset = NewsDataset(texts, labels, tokenizer, max_len)
    dataloader = DataLoader(dataset, batch_size=8, shuffle=True)

    # Split train/valid if needed
    train_loader = dataloader
    valid_loader = None

    # LSTM baseline
    lstm_model = LSTMClassifier(
        vocab_size=tokenizer.vocab_size,
        embed_dim=128,
        hidden_dim=256,
        output_dim=2
    )
    lstm_optimizer = torch.optim.Adam(lstm_model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()
    lstm_trainer = Trainer(lstm_model, train_loader, lstm_optimizer, criterion, device)
    lstm_history = lstm_trainer.train(epochs=3, valid_dataloader=valid_loader)

    # BERT baseline
    bert_model = BERTClassifier('bert-base-uncased', output_dim=2)
    bert_optimizer = AdamW(bert_model.parameters(), lr=2e-5)
    bert_trainer = Trainer(bert_model, train_loader, bert_optimizer, criterion, device)
    bert_history = bert_trainer.train(epochs=3, valid_dataloader=valid_loader)

    print("Training complete.")