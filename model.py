import torch
import torch.nn as nn
from torch.utils.data import Dataset
from transformers import BertConfig, BertModel
from tqdm import tqdm
import random


class ALDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=10, mlm_prob=0.15):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.mlm_prob = mlm_prob

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = str(self.texts[idx])

        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )

        input_ids = encoding["input_ids"].flatten()
        attention_mask = encoding["attention_mask"].flatten()
        labels = input_ids.clone()
        masked_input_ids = input_ids.clone()

        rand = torch.rand(input_ids.shape)
        mask_arr = (
            (rand < self.mlm_prob)
            * (input_ids != self.tokenizer.cls_token_id)
            * (input_ids != self.tokenizer.sep_token_id)
            * (input_ids != self.tokenizer.pad_token_id)
        )
        selection = []
        for i in range(input_ids.shape[0]):
            if mask_arr[i]:
                selection.append(i)

        for i in selection:
            if random.random() < 0.8:
                masked_input_ids[i] = self.tokenizer.mask_token_id
            elif random.random() < 0.5:
                masked_input_ids[i] = random.randint(0, self.tokenizer.vocab_size - 1)

        labels[~mask_arr] = -100

        return {
            "input_ids": masked_input_ids.flatten(),
            "attention_mask": attention_mask.flatten(),
            "labels": labels.flatten(),
        }


class ALPreTrainModel(nn.Module):
    # with MLM head
    def __init__(
        self,
        vocab_size,
        hidden_size=256,
        num_layers=4,
        num_attention_heads=4,
        max_length=10,
    ):
        super(ALPreTrainModel, self).__init__()
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_attention_heads = num_attention_heads
        self.max_length = max_length

        config = BertConfig(
            vocab_size=vocab_size,
            hidden_size=hidden_size,
            num_hidden_layers=num_layers,
            num_attention_heads=num_attention_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=max_length,
            type_vocab_size=1,  # single
            pad_token_id=0,
            position_embedding_type="absolute",
            use_cache=False,
            classifier_dropout=0.1,
        )

        self.bert = BertModel(config)
        self.mlm_head = nn.Linear(hidden_size, vocab_size)

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        prediction_scores = self.mlm_head(sequence_output)
        return prediction_scores


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
            num_attention_heads=num_attention_heads,
            intermediate_size=hidden_size * 4,
            max_position_embeddings=max_length,
            type_vocab_size=1,  # single
            pad_token_id=0,
            position_embedding_type="absolute",
            use_cache=False,
            classifier_dropout=0.1,
        )

        self.bert = BertModel(config)
        self.dropout = nn.Dropout(0.1)
        self.classifier = nn.Linear(hidden_size, num_classes)

    def load_pretrained_weights(self, pretrain_model_state_dict):
        bert_state_dict = {}
        for key, value in pretrain_model_state_dict.items():
            if key.startswith("bert."):
                new_key = key[5:]  # Remove 'bert.' prefix
                bert_state_dict[new_key] = value

        self.bert.load_state_dict(bert_state_dict, strict=False)

        print(f"loaded {len(bert_state_dict)} tensors")

    def forward(self, input_ids, attention_mask):
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)

        # Use [CLS] token representation for classification
        pooled_output = outputs.pooler_output
        pooled_output = self.dropout(pooled_output)
        logits = self.classifier(pooled_output)

        return logits

def train_mlm(model, data_loader, optimizer, scheduler, criterion, device):
    model.train()
    total_loss = 0
    total_accuracy = 0

    progress_bar = tqdm(data_loader, desc="MLM Training")
    for batch in progress_bar:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        loss = criterion(outputs.view(-1, model.vocab_size), labels.view(-1))
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

        optimizer.step()
        scheduler.step()

        mask = labels != -100
        if mask.sum() > 0:
            predictions = torch.argmax(outputs, dim=-1)
            accuracy = (predictions[mask] == labels[mask]).float().mean()
        else:
            accuracy = torch.tensor(0.0)

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


def pretrain(model, mlm_dataloader, num_epochs, learning_rate, device):
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    criterion = nn.CrossEntropyLoss(ignore_index=-100)  # Ignore -100 labels

    total_steps = len(mlm_dataloader) * num_epochs
    scheduler = torch.optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1.0, end_factor=0.0, total_iters=total_steps
    )

    best_loss = float("inf")
    best_model_state = None

    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        train_loss, train_acc = train_mlm(
            model, mlm_dataloader, optimizer, scheduler, criterion, device
        )
        print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")

        if train_loss < best_loss:
            best_loss = train_loss
            best_model_state = model.state_dict().copy()

    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    print(f"Best loss: {best_loss:.4f}")
    return model, best_model_state
