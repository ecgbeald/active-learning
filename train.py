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
from utils import process_seq
from model import ALDataset, ALClassifier, train_epoch


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


def sample_base_model_data(
    processed_df, label_0_sample_count=100, label_1_sample_count=100
):
    label_0 = processed_df[processed_df["label"] == 0].sample(
        label_0_sample_count, random_state=42
    )
    label_1 = processed_df[processed_df["label"] == 1].sample(
        label_1_sample_count, random_state=42
    )
    df_subset = pd.concat([label_0, label_1])
    subset_indices = label_0.index.union(label_1.index)
    df_rest = processed_df.drop(subset_indices).reset_index(drop=True)
    return df_subset, df_rest, subset_indices


def train_base_model(
    train_dataset,
    device,
    tokenizer,
    batch_size=64,
    epochs=10,
    model_save_path="./pretrained_classifier",
    hidden_size=512,
    num_layers=2,
    num_attention_heads=4,
    max_length=20,
):
    LEARNING_RATE = 2e-5
    WEIGHT_DECAY = 0.01
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    model = ALClassifier(
        vocab_size=tokenizer.vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_classes=2,
        num_attention_heads=num_attention_heads,
        max_length=max_length,
    )
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
    )

    criterion = nn.CrossEntropyLoss()

    total_steps = len(train_loader) * epochs
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,  # 10% warmup
        num_training_steps=total_steps,
    )
    print("Starting training...")
    print("=" * 50)

    train_losses = []
    train_accuracies = []
    # the model should not have knowledge of validation as these are "unlabelled"
    # val_losses = []
    # val_accuracies = []

    best_train_accuracy = 0
    best_model_state = None
    for epoch in range(epochs):
        print(f"\nEpoch {epoch + 1}/{epochs}")
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

    os.makedirs(model_save_path, exist_ok=True)

    torch.save(
        {
            "model_state_dict": best_model_state,
            "optimizer_state_dict": optimizer.state_dict(),
            "epoch": epochs,
            "best_train_accuracy": best_train_accuracy,
        },
        os.path.join(model_save_path, "pytorch_model.bin"),
    )

    model_config = {
        "vocab_size": tokenizer.vocab_size,
        "hidden_size": model.hidden_size,
        "num_attention_heads": model.num_attention_heads,
        "num_layers": model.num_layers,
        "num_classes": model.num_classes,
        "max_length": model.max_length,
        "model_type": "bert",
    }

    with open(os.path.join(model_save_path, "config.json"), "w") as f:
        json.dump(model_config, f, indent=2)


def copy_vocab_file(tokenizer_path, model_save_path):
    src = os.path.join(tokenizer_path, "vocab.txt")
    dst = os.path.join(model_save_path, "vocab.txt")
    with open(src, "rb") as f_src:
        with open(dst, "wb") as f_dst:
            f_dst.write(f_src.read())


def train(
    dataset,
    pre_processed=False,
    tokenizer_path="./numeric_tokenizer",
    model_save_path="./pretrained_classifier",
    window_size=5,
    debug=False,
    num_queries_al=10,
    num_samples_al=100,
    label_0_sample_count=100,
    label_1_sample_count=100,
    max_length=20,
    mini_batch_size=64,
    hidden_size=512,
    num_layers=2,
    num_attention_heads=4,
    base_model_train_epochs=10,
):
    NUM_CLASSES = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    if debug:
        timestamp = datetime.now().strftime("%m%d_%H%M%S")
        output_file = f"evaluation_results_{timestamp}.txt"

        with open(output_file, "w") as f:
            f.write(f"Evaluation started at {timestamp}\n")
            f.write(f"Using device: {device}\n")

    if not pre_processed:
        processed_df = process_seq(dataset, window_size)
    else:
        processed_df = dataset
    raw_dataset = processed_df.copy()
    df_subset, df_rest, subset_indices = sample_base_model_data(
        processed_df, label_0_sample_count, label_1_sample_count
    )

    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)
    target_labels = np.arange(NUM_CLASSES)
    X_entries = df_subset["combined"].values
    y_entries = df_subset["label"].values
    train_dataset = ALDataset(X_entries, y_entries, tokenizer, max_length=max_length)
    train_base_model(
        train_dataset,
        device,
        tokenizer,
        model_save_path=model_save_path,
        max_length=max_length,
        hidden_size=hidden_size,
        num_layers=num_layers,
        num_attention_heads=num_attention_heads,
        epochs=base_model_train_epochs,
    )
    copy_vocab_file(tokenizer_path, model_save_path)

    rest_dataset = TransformersDataset.from_arrays(
        df_rest["combined"],
        df_rest["label"],
        tokenizer,
        max_length=max_length,
        target_labels=target_labels,
    )

    train_dataset = TransformersDataset.from_arrays(
        raw_dataset["combined"],
        raw_dataset["label"],
        tokenizer,
        max_length=max_length,
        target_labels=target_labels,
    )

    indices_selected = random_initialization_balanced(rest_dataset.y, n_samples=500)

    transformer_model = TransformerModelArguments(model_save_path)

    clf_factory = TransformerBasedClassificationFactory(
        transformer_model,
        NUM_CLASSES,
        kwargs=dict(
            {
                "device": "cuda",
                "mini_batch_size": mini_batch_size,
                "class_weight": "balanced",
            }
        ),
    )
    query_strategy = EmbeddingKMeans()

    active_learner = PoolBasedActiveLearner(clf_factory, query_strategy, train_dataset)

    active_learner.initialize_data(
        indices_selected,
        train_dataset.y[indices_selected],
        indices_ignored=subset_indices,
    )

    train_acc, test_acc, f1, cm = evaluate(
        active_learner, train_dataset[indices_selected], train_dataset
    )

    for i in range(num_queries_al):
        indices_queried = active_learner.query(num_samples=num_samples_al)

        # replace with agent - get truth label
        y_true = train_dataset.y[indices_queried]

        active_learner.update(y_true)
        indices_selected = np.concatenate([indices_queried, indices_selected])

        train_acc, test_acc, f1, cm = evaluate(
            active_learner, train_dataset[indices_selected], train_dataset
        )
        if debug:
            with open(output_file, "a") as f:
                f.write("---------------\n")
                f.write(f"Iteration #{i} ({len(indices_selected)} samples)\n")
                f.write("Train accuracy: {:.5f}\n".format(train_acc))
                f.write("Test accuracy: {:.5f}\n".format(test_acc))
                f.write("Test F1 score: {:.5f}\n".format(f1))
                f.write("Confusion Matrix (Test):\n")
                f.write(str(cm) + "\n")
    return active_learner.classifier
