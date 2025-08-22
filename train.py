from transformers import AutoTokenizer, get_linear_schedule_with_warmup

from small_text import (
    TransformersDataset,
    PoolBasedActiveLearner,
    EmbeddingKMeans,
    TransformerBasedClassificationFactory,
    TransformerModelArguments,
    random_initialization_balanced,
)
from torch.utils.data import DataLoader
import torch.nn as nn
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
import pandas as pd
import json
import os
import torch
from datetime import datetime

from model import ALDataset, ALClassifier, train_epoch, validate_epoch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
timestamp = datetime.now().strftime("%m%d_%H%M%S")
output_file = f"evaluation_results_{timestamp}.txt"
with open(output_file, "w") as f:
    f.write(f"Evaluation started at {timestamp}\n")
    f.write(f"Using device: {device}\n")

print(f"Using device: {device}")


def process_alerts_csv(input_csv_path):
    """
    Process raw alert CSV files into standardised format.

    Args:
        input_csv_path (str): Path to the input CSV file in alerts_csv directory

    Returns:
        pd.DataFrame: The processed dataframe
    """

    # Read the input CSV
    print(f"Reading CSV from: {input_csv_path}")
    df = pd.read_csv(input_csv_path)
    df_1 = df[df["label"] == 1]
    df_0 = df[df["label"] == 0]
    n = 1_000_000
    df_1_sampled = df_1.sample(n=min(n, len(df_1)), random_state=42)
    df_0_sampled = df_0.sample(n=min(n, len(df_0)), random_state=42)
    df = (
        pd.concat([df_1_sampled, df_0_sampled])
        .sample(frac=1, random_state=42)
        .reset_index(drop=True)
    )
    print(f"Original shape: {df.shape}")

    # Standardize column names
    # df.rename(columns={
    #     'time': 'timestamp',
    #     'name': 'event',
    #     'host': 'machine',
    # }, inplace=True)

    # unique_events = df['event'].unique()
    # event_map = {evt: idx + 1 for idx, evt in enumerate(unique_events)}
    # df['event'] = df['event'].map(event_map)
    # unique_machines = df['machine'].unique()
    # machines_map = {mac: idx + 1 for idx, mac in enumerate(unique_machines)}
    # df['machine'] = df['machine'].map(machines_map)

    # # Create binary label column (0=normal, 1=anomalous)
    # df['label'] = df['event_label'].apply(lambda x: 0 if x == '-' else 1)

    # # Drop unnecessary columns
    # df.drop(columns=['event_label', 'time_label', 'short', 'ip'], inplace=True)

    # # Create output directory if it doesn't exist

    # df.sort_values(by=['timestamp'], inplace=True)
    # min_timestamp = df['timestamp'].min()
    # df['timestamp'] = df['timestamp'] - min_timestamp
    df["combined"] = (
        df["timestamp"].astype(str)
        + " "
        + df["event"].astype(str)
        + " "
        + df["machine"].astype(str)
    )
    print(f"Final shape: {df.shape}")
    print(f"Unique events: {len(df['event'].unique())}")
    print(f"Unique machines: {len(df['machine'].unique())}")
    print(
        f"Normal/Anomalous distribution: {sum(df['label'] == 0)}/{sum(df['label'] == 1)}"
    )
    return df


def evaluate(active_learner, train, test):
    y_pred = active_learner.classifier.predict(train)
    y_pred_test = active_learner.classifier.predict(test)

    train_acc = accuracy_score(y_pred, train.y)
    test_acc = accuracy_score(y_pred_test, test.y)
    f1 = f1_score(test.y, y_pred_test)
    cm = confusion_matrix(test.y, y_pred_test)

    print("Train accuracy: {:.5f}".format(train_acc))
    print("Test accuracy: {:.5f}".format(test_acc))
    print("Test F1 score: {:.5f}".format(f1))
    print("Confusion Matrix (Test):")
    print(cm)

    return train_acc, test_acc, f1, cm


processed_df = process_alerts_csv("microsoft/guide_alerts.csv")

raw_dataset = {
    "combined": processed_df["combined"].tolist(),
    "label": processed_df["label"].tolist(),
}
num_classes = 2

tokenizer = AutoTokenizer.from_pretrained("numeric_tokenizer")
target_labels = np.arange(num_classes)

# select 1000 samples to train pretrained model (500 from each class)
label_0 = processed_df[processed_df["label"] == 0].sample(500, random_state=42)
label_1 = processed_df[processed_df["label"] == 1].sample(500, random_state=42)

df_subset = pd.concat([label_0, label_1])

subset_indices = label_0.index.union(label_1.index)

# Remove these rows from processed_df
df_rest = processed_df.drop(subset_indices).reset_index(drop=True)

X_entries = df_subset["combined"].values
y_entries = df_subset["label"].values
train_dataset = ALDataset(X_entries, y_entries, tokenizer)

batch_size = 128
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

model = ALClassifier(
    vocab_size=tokenizer.vocab_size,
    hidden_size=512,
    num_layers=4,
    num_classes=2,
    max_length=10,
)

model = model.to(device)
EPOCHS = 20
LEARNING_RATE = 2e-5
WEIGHT_DECAY = 0.01

# Initialize optimizer and loss function
optimizer = torch.optim.AdamW(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)

criterion = nn.CrossEntropyLoss()

total_steps = len(train_loader) * EPOCHS
scheduler = get_linear_schedule_with_warmup(
    optimizer,
    num_warmup_steps=total_steps // 10,  # 10% warmup
    num_training_steps=total_steps,
)
print("Starting training...")
print("=" * 50)

# training history
train_losses = []
train_accuracies = []
# the model should not have knowledge of validation as these are "unlabelled"
# val_losses = []
# val_accuracies = []

best_train_accuracy = 0
best_model_state = None
for epoch in range(EPOCHS):
    print(f"\nEpoch {epoch + 1}/{EPOCHS}")
    print("-" * 30)

    # Training
    train_loss, train_acc = train_epoch(
        model, train_loader, optimizer, scheduler, criterion, device
    )
    # Validation
    # val_loss, val_acc, val_predictions, val_labels = validate_epoch(
    #     model, valid_loader, criterion, device
    # )
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    # val_losses.append(val_loss)
    # val_accuracies.append(val_acc)

    if train_acc > best_train_accuracy:
        best_train_accuracy = train_acc
        best_model_state = model.state_dict().copy()

    # Print epoch summary
    print(f"\nEpoch {epoch + 1} Summary:")
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    # print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    print(f"Learning Rate: {scheduler.get_last_lr()[0]:.2e}")

print("\n" + "=" * 50)
print("Training completed!")
print(f"Best training accuracy: {best_train_accuracy:.4f}")

model.load_state_dict(best_model_state)

model_save_path = "./pretrained_classifier"
os.makedirs(model_save_path, exist_ok=True)

torch.save(
    {
        "model_state_dict": best_model_state,
        "optimizer_state_dict": optimizer.state_dict(),
        "epoch": EPOCHS,
        "best_train_accuracy": best_train_accuracy,
    },
    os.path.join(model_save_path, "pytorch_model.bin"),
)

# Save model configuration
model_config = {
    "vocab_size": tokenizer.vocab_size,
    "hidden_size": model.hidden_size,
    "num_attention_heads": model.num_attention_heads,
    "num_layers": model.num_layers,
    "num_classes": model.num_classes,
    "max_length": model.max_length,
    "model_type": model.model_type,
}

with open(os.path.join(model_save_path, "config.json"), "w") as f:
    json.dump(model_config, f, indent=2)

training_history = {
    "train_losses": train_losses,
    "train_accuracies": train_accuracies,
    # 'val_losses': val_losses,
    # 'val_accuracies': val_accuracies,
    "best_train_accuracy": best_train_accuracy,
    "epochs": EPOCHS,
    "learning_rate": LEARNING_RATE,
    "batch_size": batch_size,
}

with open(os.path.join(model_save_path, "training_history.json"), "w") as f:
    json.dump(training_history, f, indent=2)


rest_dataset = TransformersDataset.from_arrays(
    df_rest["combined"],
    df_rest["label"],
    tokenizer,
    max_length=10,
    target_labels=target_labels,
)

train_dataset = TransformersDataset.from_arrays(
    raw_dataset["combined"],
    raw_dataset["label"],
    tokenizer,
    max_length=10,
    target_labels=target_labels,
)

indices_selected = random_initialization_balanced(rest_dataset.y, n_samples=500)

transformer_model = TransformerModelArguments(model_save_path)

clf_factory = TransformerBasedClassificationFactory(
    transformer_model,
    num_classes,
    kwargs=dict({"device": "cuda", "mini_batch_size": 256, "class_weight": "balanced"}),
)
query_strategy = EmbeddingKMeans()

active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, train_dataset)

active_learner.initialize_data(
    indices_selected, train_dataset.y[indices_selected], indices_ignored=subset_indices
)

num_queries = 50
num_samples = 100

train_acc, test_acc, f1, cm = evaluate(
    active_learner, train_dataset[indices_selected], train_dataset
)

for i in range(num_queries):
    indices_queried = active_learner.query(num_samples=num_samples)

    # replace with agent - get truth label
    y_true = train_dataset.y[indices_queried]

    active_learner.update(y_true)
    indices_selected = np.concatenate([indices_queried, indices_selected])

    train_acc, test_acc, f1, cm = evaluate(
        active_learner, train_dataset[indices_selected], train_dataset
    )

    with open(output_file, "a") as f:
        f.write("---------------\n")
        f.write(f"Iteration #{i} ({len(indices_selected)} samples)\n")
        f.write("Train accuracy: {:.5f}\n".format(train_acc))
        f.write("Test accuracy: {:.5f}\n".format(test_acc))
        f.write("Test F1 score: {:.5f}\n".format(f1))
        f.write("Confusion Matrix (Test):\n")
        f.write(str(cm) + "\n")
