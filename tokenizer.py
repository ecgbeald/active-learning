from utils import process_seq
from transformers import PreTrainedTokenizerFast
from tokenizers import Tokenizer, models, pre_tokenizers, trainers, normalizers
import json
import os


def generate_bert_vocab_file(tokenizer, output_path="./numeric_tokenizer"):
    vocab = tokenizer.get_vocab()

    sorted_vocab = sorted(vocab.items(), key=lambda x: x[1])
    vocab_file = os.path.join(output_path, "vocab.txt")
    os.makedirs(os.path.dirname(vocab_file), exist_ok=True)
    with open(vocab_file, "w", encoding="utf-8") as f:
        for token, _ in sorted_vocab:
            f.write(f"{token}\n")

    json_path = vocab_file.replace(".txt", ".json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(vocab, f, ensure_ascii=False, indent=2)

    return vocab_file, json_path


def create_numeric_tokenizer_with_priority(processed_df, vocab_size=10000, tokenize_param=False):
    tokenizer = Tokenizer(models.WordPiece())
    special_tokens = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"]

    tokenizer.normalizer = normalizers.Sequence([])

    special_token_pattern = r"\[CLS\]|\[SEP\]|\[PAD\]|\[UNK\]|\[MASK\]"

    tokenizer.pre_tokenizer = pre_tokenizers.Sequence(
        [
            pre_tokenizers.Split(pattern=special_token_pattern, behavior="isolated"),
            pre_tokenizers.Whitespace(),
        ]
    )

    event_counts = processed_df["event"].value_counts()
    if tokenize_param:
        param_counts = processed_df["param"].value_counts()

    reserved_tokens = (
        len(special_tokens) + 1000
    )  # Reserve 1000 for general number tokens
    available_for_priority = vocab_size - reserved_tokens

    total_unique = len(event_counts)
    if tokenize_param:
        tokenize_param += len(param_counts)

    if total_unique <= available_for_priority:
        top_events = event_counts.index.tolist()
        if tokenize_param:
            top_params = param_counts.index.tolist()
    else:
        event_ratio = len(event_counts) / total_unique
        top_events_count = int(available_for_priority * event_ratio)
        if tokenize_param:
            top_params_count = available_for_priority - top_events_count

        top_events = event_counts.head(top_events_count).index.tolist()
        if tokenize_param:
            top_params = param_counts.head(top_params_count).index.tolist()

    priority_tokens = []
    priority_tokens.extend([str(event) for event in top_events])
    if tokenize_param:
        priority_tokens.extend([str(param) for param in top_params])

    priority_tokens = sorted(list(set(priority_tokens)))
    print(f"Created {len(priority_tokens)} priority tokens")

    all_special_tokens = special_tokens + priority_tokens
    trainer = trainers.WordPieceTrainer(
        vocab_size=vocab_size,
        special_tokens=all_special_tokens,
        min_frequency=1,
        show_progress=True,
    )

    return tokenizer, trainer, priority_tokens, top_events


def preprocess_text_for_special_tokens(text_series):
    processed_texts = []
    for text in text_series:
        processed_text = (
            text.replace("[SEP]", " [SEP] ")
            .replace("[CLS]", " [CLS] ")
            .replace("[MASK]", " [MASK] ")
        )
        processed_text = " ".join(processed_text.split())
        processed_texts.append(processed_text)
    return processed_texts


def tokenize(df, window_size=5, save_path="./numeric_tokenizer", vocab_size=10000):
    processed_df = process_seq(df, window_size)
    tokenize_alt(processed_df, df, save_path, vocab_size)


def tokenize_alt(processed_df, events, save_path="./numeric_tokenizer", vocab_size=10000, tokenize_param=False):
    has_special_tokens = any(
        "[SEP]" in text or "[CLS]" in text or "[MASK]" in text
        for text in processed_df["combined"]
    )
    if has_special_tokens:
        training_texts = preprocess_text_for_special_tokens(processed_df["combined"])
    else:
        training_texts = processed_df["combined"].tolist()

    tokenizer, trainer, _, _ = create_numeric_tokenizer_with_priority(
        events, vocab_size=vocab_size, tokenize_param=tokenize_param
    )

    tokenizer.train_from_iterator(training_texts, trainer)

    hf_tokenizer = PreTrainedTokenizerFast(tokenizer_object=tokenizer)

    hf_tokenizer.add_special_tokens(
        {
            "pad_token": "[PAD]",
            "unk_token": "[UNK]",
            "cls_token": "[CLS]",
            "sep_token": "[SEP]",
            "mask_token": "[MASK]",
        }
    )

    if hf_tokenizer.mask_token_id is None:
        mask_token_id = hf_tokenizer.convert_tokens_to_ids("[MASK]")
        if mask_token_id is not None and mask_token_id != hf_tokenizer.unk_token_id:
            hf_tokenizer.mask_token_id = mask_token_id

    hf_tokenizer.save_pretrained(save_path)
    _, _ = generate_bert_vocab_file(hf_tokenizer, save_path)
