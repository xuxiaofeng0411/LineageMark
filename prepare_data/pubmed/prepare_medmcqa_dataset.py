import argparse
import json
import os
import shutil
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


DEFAULT_PARQUET_PATH = "/root/autodl-tmp/datasets/train-00000-of-00001.parquet"
DEFAULT_MODEL_PATH = "/root/autodl-tmp/models/facebook/opt-125m"
DEFAULT_OUTPUT_DIR = "/root/autodl-tmp/datasets/processed_medmcqa_35k"

ANSWER_LABELS = ["A", "B", "C", "D"]
OPTION_COLUMNS = ["opa", "opb", "opc", "opd"]


def setup_environment():
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def parse_args():
    parser = argparse.ArgumentParser(
        description="Prepare a 35k MedMCQA subset for causal LM full fine-tuning."
    )
    parser.add_argument("--parquet_path", type=str, default=DEFAULT_PARQUET_PATH)
    parser.add_argument("--model_path", type=str, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output_dir", type=str, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--sample_size", type=int, default=35_000)
    parser.add_argument("--train_ratio", type=float, default=0.9)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument("--min_question_chars", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_proc", type=int, default=1)
    parser.add_argument(
        "--use_explanation",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include MedMCQA explanation text after the correct answer.",
    )
    parser.add_argument(
        "--include_metadata",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Include subject and topic fields when available.",
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
    if args.min_question_chars < 0:
        raise ValueError("--min_question_chars must be non-negative")


def clean_text(value):
    if value is None:
        return ""
    return " ".join(str(value).strip().split())


def parse_correct_option(value):
    try:
        answer_index = int(value)
    except (TypeError, ValueError):
        return None

    # araag2/MedMCQA source stores cop as 1-based: 1=A, 2=B, 3=C, 4=D.
    if 1 <= answer_index <= 4:
        return answer_index - 1
    return None


def load_and_sample_medmcqa(args):
    print("=" * 60)
    print("Loading MedMCQA parquet")
    print("=" * 60)
    print(f"parquet_path: {args.parquet_path}")

    dataset = load_dataset(
        "parquet",
        data_files=args.parquet_path,
        split="train",
    )
    print(f"raw rows: {len(dataset)}")
    print(f"raw columns: {dataset.column_names}")

    required_columns = {"question", "cop", *OPTION_COLUMNS}
    missing_columns = required_columns.difference(dataset.column_names)
    if missing_columns:
        raise ValueError(f"Missing required columns: {sorted(missing_columns)}")

    def has_valid_medmcqa_fields(example):
        question = clean_text(example.get("question"))
        options = [clean_text(example.get(column)) for column in OPTION_COLUMNS]
        correct_option = parse_correct_option(example.get("cop"))
        return (
            len(question) >= args.min_question_chars
            and all(options)
            and correct_option is not None
        )

    dataset = dataset.filter(
        has_valid_medmcqa_fields,
        num_proc=args.num_proc,
        desc="Filtering invalid MedMCQA examples",
    )
    print(f"rows after filtering: {len(dataset)}")

    if len(dataset) < args.sample_size:
        raise ValueError(
            f"Not enough valid MedMCQA examples: requested {args.sample_size}, got {len(dataset)}"
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
            question = clean_text(example.get("question"))
            options = [clean_text(example.get(column)) for column in OPTION_COLUMNS]
            correct_option = parse_correct_option(example.get("cop"))
            answer_label = ANSWER_LABELS[correct_option]
            answer_text = options[correct_option]
            explanation = clean_text(example.get("exp"))
            subject = clean_text(example.get("subject_name"))
            topic = clean_text(example.get("topic_name"))

            parts = []
            if args.include_metadata and subject:
                parts.extend(["Subject:", subject, ""])
            if args.include_metadata and topic:
                parts.extend(["Topic:", topic, ""])

            parts.extend(
                [
                    "Question:",
                    question,
                    "",
                    "Options:",
                    f"A. {options[0]}",
                    f"B. {options[1]}",
                    f"C. {options[2]}",
                    f"D. {options[3]}",
                    "",
                    "Answer:",
                    f"{answer_label}. {answer_text}",
                ]
            )
            if args.use_explanation and explanation:
                parts.extend(["", "Explanation:", explanation])

            texts.append("\n".join(parts) + eos)

        return {"text": texts}

    return sampled.map(
        format_batch,
        batched=True,
        remove_columns=sampled.column_names,
        num_proc=args.num_proc,
        desc="Formatting MedMCQA examples",
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
        desc="Tokenizing MedMCQA examples",
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
        "min_question_chars": args.min_question_chars,
        "seed": args.seed,
        "use_explanation": args.use_explanation,
        "include_metadata": args.include_metadata,
        "columns": ["input_ids", "attention_mask", "labels"],
        "format": "causal_language_modeling",
        "task_format": "medmcqa_multiple_choice_qa",
    }

    with open(output_dir / "stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2, ensure_ascii=False)

    print("=" * 60)
    print("Saved processed MedMCQA dataset")
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

    sampled = load_and_sample_medmcqa(args)
    formatted = format_dataset(sampled, tokenizer, args)
    train_dataset, val_dataset = tokenize_and_split(formatted, tokenizer, args)

    show_preview(train_dataset, tokenizer)
    save_outputs(train_dataset, val_dataset, tokenizer, args)


if __name__ == "__main__":
    main()
