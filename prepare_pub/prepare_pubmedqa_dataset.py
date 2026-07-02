import argparse
import json
import os
import shutil
from pathlib import Path
from datasets import load_dataset
from transformers import AutoTokenizer

DEFAULT_PARQUET_PATH = "/root/autodl-tmp/datasets/train-00000-of-00001.parquet"
DEFAULT_MODEL_PATH = "/root/autodl-tmp/models/facebook/opt-125m"
DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/datasets/processed_pubmed3_35k"

def setup_environment():
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a 35k PubMedQA subset for causal LM full fine-tuning."
    )
    parser.add_argument("--parquet_path", type=str, default=DEFAULT_PARQUET_PATH)
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample_size", type=int, default=35_000)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--min_context_chars", type=int, default=100)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument(
        "--use_long_answer",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include PubMedQA long_answer as an explanation after the final decision.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete output_dir first if it already exists.",
    )
    return parser.parse_args()

def require_valid_args(args):
    parquet_path = Path(args.parquet_path)
    model_path = Path(args.model_path)
    if not parquet_path.exists():
        raise FileNotFoundError(f"Parquet file does not exist: {parquet_path}")
    if not model_path.exists():
        raise FileNotFoundError(f"Tokenizer/model path does not exist: {model_path}")
    if args.sample_size <= 0:
        raise ValueError("--sample_size must be positive")
    if not 0.0 < args.train_ratio < 1.0:
        raise ValueError("--train_ratio must be between 0 and 1")
    if args.max_length <= 0:
        raise ValueError("--max_length must be positive")
    if args.min_context_chars < 0:
        raise ValueError("--min_context_chars must be non-negative")

def first_present(example, names):
    for name in names:
        if name in example:
            value = example.get(name)
            if value is not None:
                return value
    return None

def clean_text(value):
    if value is None:
        return ""
    return " ".join(str(value).strip().split())

def normalize_context(context):
    if context is None:
        return ""
    if isinstance(context, dict):
        contexts = context.get("contexts") or context.get("CONTEXTS") or []
        labels = context.get("labels") or context.get("LABELS") or []
        if isinstance(contexts, str):
            contexts = [contexts]
        if isinstance(labels, str):
            labels = [labels]
        pieces = []
        for idx, sentence in enumerate(contexts):
            sentence = clean_text(sentence)
            if not sentence:
                continue
            label = clean_text(labels[idx]) if idx < len(labels) else ""
            if label:
                pieces.append(f"{label}: {sentence}")
            else:
                pieces.append(sentence)
        return "\n".join(pieces)
    if isinstance(context, (list, tuple)):
        return "\n".join(clean_text(item) for item in context if clean_text(item))
    return clean_text(context)

def get_field(example, names):
    value = first_present(example, names)
    if names and names[0].lower() == "context":
        return normalize_context(value)
    return clean_text(value)

def load_and_sample_pubmedqa(args):
    print("=" * 60)
    print("Loading PubMedQA parquet")
    print("=" * 60)
    print(f"parquet_path: {args.parquet_path}")
    dataset = load_dataset(
        "parquet",
        data_files=args.parquet_path,
        split="train",
    )
    print(f"raw rows: {len(dataset)}")
    print(f"raw columns: {dataset.column_names}")
    required_column_groups = {
        "question": ["question", "QUESTION"],
        "context": ["context", "contexts", "CONTEXTS"],
        "final_decision": ["final_decision", "FINAL_DECISION"],
    }
    for field_name, candidates in required_column_groups.items():
        if not any(column in dataset.column_names for column in candidates):
            raise ValueError(
                f"Missing required {field_name} column. "
                f"Expected one of {candidates}, got {dataset.column_names}"
            )
    def has_valid_pubmedqa_fields(example):
        question = get_field(example, ["question", "QUESTION"])
        context = get_field(example, ["context", "contexts", "CONTEXTS"])
        final_decision = get_field(example, ["final_decision", "FINAL_DECISION"])
        return (
            len(question) > 0
            and len(context) >= args.min_context_chars
            and final_decision.lower() in {"yes", "no", "maybe"}
        )
    dataset = dataset.filter(
        has_valid_pubmedqa_fields,
        num_proc=args.num_proc,
        desc="Filtering invalid PubMedQA examples",
    )
    print(f"rows after filtering: {len(dataset)}")
    if len(dataset) < args.sample_size:
        raise ValueError(
            f"Not enough valid PubMedQA examples: requested {args.sample_size}, got {len(dataset)}"
        )
    sampled = dataset.shuffle(seed=args.seed).select(range(args.sample_size))
    print(f"sampled rows: {len(sampled)}")
    return sampled

def format_dataset(sampled, tokenizer, args):
    eos = tokenizer.eos_token or ""
    def format_batch(examples):
        texts = []
        batch_size = len(next(iter(examples.values())))
        for idx in range(batch_size):
            example = {column: values[idx] for column, values in examples.items()}
            question = get_field(example, ["question", "QUESTION"])
            context = get_field(example, ["context", "contexts", "CONTEXTS"])
            final_decision = get_field(example, ["final_decision", "FINAL_DECISION"]).lower()
            long_answer = get_field(example, ["long_answer", "LONG_ANSWER"])
            parts = [
                "Context:",
                context,
                "",
                "Question:",
                question,
                "",
                "Answer:",
                final_decision,
            ]
            if args.use_long_answer and long_answer:
                parts.extend(["", "Explanation:", long_answer])
            texts.append("\n".join(parts) + eos)
        return {"text": texts}
    return sampled.map(
        format_batch,
        batched=True,
        remove_columns=sampled.column_names,
        num_proc=args.num_proc,
        desc="Formatting PubMedQA examples",
    )

def tokenize_and_split(formatted, tokenizer, args):
    split = formatted.train_test_split(
        test_size=1.0 - args.train_ratio,
        seed=args.seed,
    )
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
    tokenized = split.map(
        tokenize_batch,
        batched=True,
        remove_columns=["text"],
        num_proc=args.num_proc,
        desc="Tokenizing PubMedQA examples",
    )
    train_dataset = tokenized["train"]
    val_dataset = tokenized["test"]
    train_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    val_dataset.set_format(
        type="torch",
        columns=["input_ids", "attention_mask", "labels"],
    )
    return train_dataset, val_dataset

def save_outputs(train_dataset, val_dataset, tokenizer, args):
    output_dir = Path(args.output_dir)
    if output_dir.exists():
        if not args.overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_dir}. "
                "Pass --overwrite to replace it."
            )
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    train_path = output_dir / "train"
    val_path = output_dir / "val"
    train_dataset.save_to_disk(str(train_path))
    val_dataset.save_to_disk(str(val_path))
    tokenizer.save_pretrained(str(output_dir))
    stats = {
        "source_parquet": args.parquet_path,
        "model_path": args.model_path,
        "sample_size": args.sample_size,
        "train_samples": len(train_dataset),
        "val_samples": len(val_dataset),
        "train_ratio": args.train_ratio,
        "max_length": args.max_length,
        "min_context_chars": args.min_context_chars,
        "seed": args.seed,
        "use_long_answer": args.use_long_answer,
        "columns": ["input_ids", "attention_mask", "labels"],
        "format": "causal_language_modeling",
        "task_format": "pubmedqa_context_question_answer",
    }
    with open(output_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)
    print("=" * 60)
    print("Saved processed PubMedQA dataset")
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
        input_ids = train_dataset[idx]["input_ids"]
        text = tokenizer.decode(input_ids, skip_special_tokens=True)
        print(f"\n--- example {idx + 1} ---")
        print(text[:1000])

def main():
    setup_environment()
    args = parse_args()
    require_valid_args(args)
    print("=" * 60)
    print("Loading tokenizer")
    print("=" * 60)
    print(f"model_path: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    sampled = load_and_sample_pubmedqa(args)
    formatted = format_dataset(sampled, tokenizer, args)
    train_dataset, val_dataset = tokenize_and_split(formatted, tokenizer, args)
    show_preview(train_dataset, tokenizer)
    save_outputs(train_dataset, val_dataset, tokenizer, args)

if __name__ == "__main__":
    main()
