from utils import process_csv
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, processors
import json
import os


def generate_bert_vocab_file(tokenizer, output_path="./numeric_tokenizer/vocab.txt"):
    """
    Generate a vocabulary file compatible with BERT models.
    BERT expects a vocab.txt file with one token per line, ordered by token ID.
    """
    vocab = tokenizer.get_vocab()
    print(f"Vocabulary size: {len(vocab)}")

    # Sort tokens by their IDs (BERT expects tokens in ID order)
    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for token, token_id in sorted_vocab:
            f.write(f"{token}\n")

    json_path = output_path.replace(".txt", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    return output_path, json_path


def create_numeric_tokenizer_with_priority(processed_df, vocab_size=10000):
    tokenizer = Tokenizer(models.WordPiece())
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]"]
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()

    # Get frequency counts for events and machines - prioritise most frequent ones
    event_counts = processed_df["event"].value_counts()
    machine_counts = processed_df["machine"].value_counts()

    # print(f"Found {len(event_counts)} unique events and {len(machine_counts)} unique machines")
    # print(f"Target vocabulary size: {vocab_size}")

    reserved_tokens = (
        len(special_tokens) + 1000
    )  # Reserve 1000 for general number tokens
    available_for_priority = vocab_size - reserved_tokens

    total_unique = len(event_counts) + len(machine_counts)

    if total_unique <= available_for_priority:
        # We can include all events and machines
        top_events = event_counts.index.tolist()
        top_machines = machine_counts.index.tolist()
    else:
        # Need to select top most frequent ones
        # Split available space between events and machines proportionally
        event_ratio = len(event_counts) / total_unique
        top_events_count = int(available_for_priority * event_ratio)
        top_machines_count = available_for_priority - top_events_count

        top_events = event_counts.head(top_events_count).index.tolist()
        top_machines = machine_counts.head(top_machines_count).index.tolist()

    priority_tokens = []
    priority_tokens.extend([str(event) for event in top_events])
    priority_tokens.extend([str(machine) for machine in top_machines])

    priority_tokens = sorted(list(set(priority_tokens)))
    print(f"Created {len(priority_tokens)} priority tokens")

    all_special_tokens = special_tokens + priority_tokens
    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size, special_tokens=all_special_tokens, min_frequency=1
    )

    return tokenizer, trainer, priority_tokens, top_events, top_machines


processed_df = process_csv("microsoft/guide_alerts.csv")

tokenizer, trainer, priority_tokens, top_events, top_machines = (
    create_numeric_tokenizer_with_priority(processed_df, vocab_size=10000)
)

unique_combined_texts = processed_df["combined"].unique()

tokenizer.train_from_iterator(unique_combined_texts, trainer)

tokenizer.post_processor = processors.TemplateProcessing(
    single="[CLS] $A [SEP]",
    pair="[CLS] $A [SEP] $B:1 [SEP]:1",
    special_tokens=[
        ("[CLS]", 2),
        ("[SEP]", 3),
    ],
)

hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)
hf_tokenizer.add_special_tokens(
    {
        "pad_token": "[PAD]",
        "unk_token": "[UNK]",
        "cls_token": "[CLS]",
        "sep_token": "[SEP]",
    }
)
hf_tokenizer.save_pretrained("./numeric_tokenizer")

_, _ = generate_bert_vocab_file(hf_tokenizer)
