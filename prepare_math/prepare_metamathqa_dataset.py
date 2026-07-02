import argparse
import json
import os
import random
import shutil
from pathlib import Path
from datasets import Dataset
from transformers import AutoTokenizer

DEFAULT_JSON_PATH = "/root/autodl-tmp/datasets/MetaMathQA-395K.json"
DEFAULT_MODEL_PATH = "/root/autodl-tmp/models/facebook/opt-125m"
DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/datasets/processed_math1_35k"

def setup_environment():
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a 35k MetaMathQA subset for causal LM full fine-tuning."
    )
    parser.add_argument("--json_path", type=str, default=DEFAULT_JSON_PATH)
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample_size", type=int, default=35_000)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--min_query_chars", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()

def require_valid_args(args):
    if not Path(args.json_path).exists():
        raise FileNotFoundError(f"JSON file does not exist: {args.json_path}")
    if not Path(args.model_path).exists():
        raise FileNotFoundError(f"Tokenizer/model path does not exist: {args.model_path}")
    if args.sample_size <= 0:
        raise ValueError("--sample_size must be positive")
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train_ratio must be between 0 and 1")
    if args.max_length <= 0:
        raise ValueError("--max_length must be positive")

def clean_text(value):
    if value is None:
        return ""
    return " ".join(str(value).strip().split())

def load_complete_json_array_records(path):
    text = Path(path).read_text(encoding="utf-8")
    decoder = json.JSONDecoder()
    pos = 0
    total = len(text)
    records = []
    decode_error = None
    while pos < total and text[pos].isspace():
        pos += 1
    if pos >= total or text[pos] != "[":
        raise ValueError(f"Expected a JSON array in {path}")
    pos += 1
    while pos < total:
        while pos < total and text[pos].isspace():
            pos += 1
        if pos < total and text[pos] == ",":
            pos += 1
            continue
        if pos < total and text[pos] == "]":
            pos += 1
            break
        try:
            record, pos = decoder.raw_decode(text, pos)
        except json.JSONDecodeError as exc:
            decode_error = f"{exc.msg} at char {exc.pos}"
            break
        if isinstance(record, dict):
            records.append(record)
    trailing = text[pos:].strip()
    is_truncated = decode_error is not None or (trailing and trailing != "]")
    return records, is_truncated, decode_error

def load_and_sample(args):
    print("=" * 60)
    print("Loading MetaMathQA JSON")
    print("=" * 60)
    print(f"json_path: {args.json_path}")
    records, is_truncated, decode_error = load_complete_json_array_records(args.json_path)
    print(f"complete records parsed: {len(records)}")
    print(f"is_truncated_or_invalid_tail: {is_truncated}")
    if decode_error:
        print(f"decode_error: {decode_error}")
    valid = []
    for record in records:
        query = clean_text(record.get("query"))
        response = clean_text(record.get("response"))
        if len(query) >= args.min_query_chars and response:
            valid.append(
                {
                    "query": query,
                    "response": response,
                    "type": clean_text(record.get("type")),
                }
            )
    print(f"valid rows: {len(valid)}")
    if len(valid) < args.sample_size:
        raise ValueError(f"Not enough valid rows: requested {args.sample_size}, got {len(valid)}")
    rng = random.Random(args.seed)
    rng.shuffle(valid)
    sampled = Dataset.from_list(valid[: args.sample_size])
    print(f"sampled rows: {len(sampled)}")
    return sampled, {"is_truncated": is_truncated, "decode_error": decode_error}

def format_dataset(sampled, tokenizer, args):
    eos = tokenizer.eos_token or ""
    def format_batch(examples):
        texts = []
        for query, response, source_type in zip(examples["query"], examples["response"], examples["type"]):
            parts = []
            if source_type:
                parts.extend(["Source Type:", source_type, ""])
            parts.extend(["Problem:", query, "", "Solution:", response])
            texts.append("\n".join(parts) + eos)
        return {"text": texts}
    return sampled.map(format_batch, batched=True, remove_columns=sampled.column_names, desc="Formatting MetaMathQA examples")

def tokenize_and_split(formatted, tokenizer, args):
    split = formatted.train_test_split(test_size=1.0 - args.train_ratio, seed=args.seed)
    def tokenize_batch(examples):
        tokenized = tokenizer(
            examples["text"],
            truncation=True,
            padding="max_length",
            max_length=args.max_length,
            return_tensors=None,
        )
        tokenized["labels"] = [input_ids.copy() for input_ids in tokenized["input_ids"]]
        return tokenized
    tokenized = split.map(tokenize_batch, batched=True, remove_columns=["text"], desc="Tokenizing MetaMathQA examples")
    train_dataset = tokenized["train"]
    val_dataset = tokenized["test"]
    train_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    val_dataset.set_format(type="torch", columns=["input_ids", "attention_mask", "labels"])
    return train_dataset, val_dataset

def save_outputs(train_dataset, val_dataset, tokenizer, source_status, args):
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Output directory already exists: {output_dir}. Pass --overwrite to replace it.")
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train"
    val_path = output_dir / "val"
    train_dataset.save_to_disk(str(train_path))
    val_dataset.save_to_disk(str(val_path))
    tokenizer.save_pretrained(str(output_dir))
    stats = {
        "source_json": args.json_path,
        "model_path": args.model_path,
        "sample_size": args.sample_size,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "train_ratio": args.train_ratio,
        "max_length": args.max_length,
        "min_query_chars": args.min_query_chars,
        "seed": args.seed,
        "source_status": source_status,
        "columns": ["input_ids", "attention_mask", "labels"],
        "format": "causal_language_modeling",
        "task_format": "metamathqa_problem_solution",
    }
    with open(output_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print("=" * 60)
    print("Saved processed MetaMathQA dataset")
    print("=" * 60)
    print(f"output_dir: {output_dir}")
    print(f"train: {train_path} ({len(train_dataset)} samples)")
    print(f"val:   {val_path} ({len(val_dataset)} samples)")
    print(f"stats: {output_dir / 'stats.json'}")

def show_preview(train_dataset, tokenizer, num_examples=2):
    print("=" * 60)
    print("Preview")
    print("=" * 60)
    for idx in range(min(num_examples, len(train_dataset))):
        print(f"\n--- example {idx + 1} ---")
        print(tokenizer.decode(train_dataset[idx]["input_ids"], skip_special_tokens=True)[:1000])

def main():
    setup_environment()
    args = parse_args()
    require_valid_args(args)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    sampled, source_status = load_and_sample(args)
    formatted = format_dataset(sampled, tokenizer, args)
    train_dataset, val_dataset = tokenize_and_split(formatted, tokenizer, args)
    show_preview(train_dataset, tokenizer)
    save_outputs(train_dataset, val_dataset, tokenizer, source_status, args)

if __name__ == "__main__":
    main()
