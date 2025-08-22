import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertConfig, BertModel
from tqdm import tqdm


class ALDataset(Dataset):
    def __init__(self, texts, labels, tokenizer, max_length=10):
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])
        label = self.labels[idx]

        # Tokenize the text
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class ALClassifier(nn.Module):
    def __init__(
        self,
        vocab_size,
        hidden_size=256,
        num_layers=4,
        num_classes=2,
        num_attention_heads=4,
        max_length=10,
    ):
        super(ALClassifier, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_classes = num_classes
        self.num_attention_heads = num_attention_heads
        self.max_length = max_length

        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=4,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=max_length,
            type_vocab_size=1,
            pad_token_id=0,
            position_embedding_type="absolute",
            use_cache=True,
            classifier_dropout=0.1,
        )

        self.bert = BertModel(config)

        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation for classification
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits


def calculate_accuracy(predictions, labels):
    pred_classes = torch.argmax(predictions, dim=1)
    return torch.sum(pred_classes == labels).double() / len(labels)


def train_epoch(model, data_loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    total_accuracy = 0

    progress_bar = tqdm(data_loader, desc="Training")
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs, labels)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        accuracy = calculate_accuracy(outputs, labels)
        total_loss += loss.item()
        total_accuracy += accuracy.item()

        progress_bar.set_postfix(
            {
                "loss": f"{loss.item():.4f}",
                "acc": f"{accuracy.item():.4f}",
                "lr": f"{scheduler.get_last_lr()[0]:.2e}",
            }
        )

    avg_loss = total_loss / len(data_loader)
    avg_accuracy = total_accuracy / len(data_loader)

    return avg_loss, avg_accuracy


def validate_epoch(model, data_loader, criterion, device):
    model.eval()
    total_loss = 0
    total_accuracy = 0
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        progress_bar = tqdm(data_loader, desc="Validation")

        for batch in progress_bar:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            loss = criterion(outputs, labels)

            accuracy = calculate_accuracy(outputs, labels)
            total_loss += loss.item()
            total_accuracy += accuracy.item()

            predictions = torch.argmax(outputs, dim=1)
            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

            progress_bar.set_postfix(
                {"loss": f"{loss.item():.4f}", "acc": f"{accuracy.item():.4f}"}
            )

    avg_loss = total_loss / len(data_loader)
    avg_accuracy = total_accuracy / len(data_loader)

    return avg_loss, avg_accuracy, all_predictions, all_labels
