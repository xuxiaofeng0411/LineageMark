import argparse
import gc
import json
import os
import time
from dataclasses import dataclass
import torch
import torch.nn.functional as F
from datasets import DownloadConfig, load_dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DATASET_CACHE_DIR = "/root/autodl-tmp/datasets"
DEFAULT_OUTPUT_JSON = (
    "/root/autodl-tmp/project12-matrix-optimize-nomark/outputs/"
    "multi_stage_zero_shot_results.json"
)
MODEL_SPECS = [
    ("original", "/root/autodl-tmp/models/facebook/opt-125m"),
    ("watermark_1", "/root/autodl-tmp/models/facebook/opt-125m-mark1-nomark"),
    ("finetune_1_pubmed", "/root/autodl-tmp/models/opt-125m-mark1-nomark-ft-pubmed"),
    ("watermark_2", "/root/autodl-tmp/models/facebook/opt-125m-mark2-nomark"),
    ("finetune_2_pubmedqa", "/root/autodl-tmp/models/opt-125m-mark2-nomark-ft-pubmed"),
    ("watermark_3", "/root/autodl-tmp/models/facebook/opt-125m-mark3-nomark"),
    ("finetune_3_medcqa", "/root/autodl-tmp/models/opt-125m-mark3-nomark-ft-pubmed"),
    ("watermark_4", "/root/autodl-tmp/models/facebook/opt-125m-mark4-nomark"),
]

@dataclass
class ChoiceExample:
    prompt: str
    choices: list[str]
    label: int

@dataclass
class TaskResult:
    model_name: str
    model_path: str
    task_name: str
    accuracy: float
    correct: int
    total: int
    elapsed_sec: float

def setup_environment():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
    os.environ["HF_DATASETS_OFFLINE"] = "1"
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    torch.backends.cuda.matmul.allow_tf32 = True

def format_time(seconds):
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"

def load_cached_dataset(path, name=None, split="validation"):
    attempts = [(None, True), (DATASET_CACHE_DIR, True)]
    errors = []
    for cache_dir, local_only in attempts:
        kwargs = {"path": path, "split": split}
        if name is not None:
            kwargs["name"] = name
        if cache_dir is not None:
            kwargs["cache_dir"] = cache_dir
        if local_only:
            kwargs["download_config"] = DownloadConfig(local_files_only=True)
        try:
            dataset = load_dataset(**kwargs)
            print(
                f"Loaded dataset path={path}, name={name}, split={split}, "
                f"cache_dir={cache_dir or 'default'}, local_only={local_only}, "
                f"samples={len(dataset)}"
            )
            return dataset
        except Exception as exc:
            errors.append(
                f"cache_dir={cache_dir or 'default'}, local_only={local_only}: "
                f"{type(exc).__name__}: {exc}"
            )
    raise RuntimeError(f"Could not load dataset {path}/{name}/{split}:\n" + "\n".join(errors))

def normalize_choice(choice):
    choice = str(choice).strip()
    if not choice:
        return choice
    return " " + choice

def build_piqa_examples(limit=None):
    dataset = load_cached_dataset("piqa", split="validation")
    examples = []
    for sample in dataset:
        examples.append(
            ChoiceExample(
                prompt=f"Question: {sample['goal'].strip()}\nAnswer:",
                choices=[normalize_choice(sample["sol1"]), normalize_choice(sample["sol2"])],
                label=int(sample["label"]),
            )
        )
        if limit is not None and len(examples) >= limit:
            break
    return examples

def build_hellaswag_examples(limit=None):
    dataset = load_cached_dataset("hellaswag", split="validation")
    examples = []
    for sample in dataset:
        ctx = sample.get("ctx")
        if ctx is None:
            ctx = (sample.get("ctx_a", "") + " " + sample.get("ctx_b", "")).strip()
        examples.append(
            ChoiceExample(
                prompt=ctx.strip(),
                choices=[normalize_choice(choice) for choice in sample["endings"]],
                label=int(sample["label"]),
            )
        )
        if limit is not None and len(examples) >= limit:
            break
    return examples

def build_winogrande_examples(limit=None):
    dataset = load_cached_dataset("winogrande", name="winogrande_xl", split="validation")
    examples = []
    for sample in dataset:
        sentence = sample["sentence"]
        if "_" in sentence:
            prefix, suffix = sentence.split("_", 1)
        else:
            prefix, suffix = sentence, ""
        examples.append(
            ChoiceExample(
                prompt=prefix,
                choices=[
                    normalize_choice(sample["option1"] + suffix),
                    normalize_choice(sample["option2"] + suffix),
                ],
                label=int(sample["answer"]) - 1,
            )
        )
        if limit is not None and len(examples) >= limit:
            break
    return examples

def load_task_examples(task_name, limit=None):
    if task_name == "piqa":
        return build_piqa_examples(limit)
    if task_name == "hellaswag":
        return build_hellaswag_examples(limit)
    if task_name == "winogrande":
        return build_winogrande_examples(limit)
    raise ValueError(f"Unsupported task: {task_name}")

def load_model_and_tokenizer(model_name, model_path):
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model path does not exist for {model_name}: {model_path}")

    print("\n" + "=" * 80)
    print(f"Loading model: {model_name}")
    print("=" * 80)
    print(f"path: {model_path}")
    tokenizer = AutoTokenizer.from_pretrained(model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=False,
    )
    model.eval()
    model.config.use_cache = False
    model.config.pad_token_id = tokenizer.pad_token_id
    print(f"parameter dtype: {next(model.parameters()).dtype}")
    print(f"device: {next(model.parameters()).device}")
    return model, tokenizer

def encode_choice(tokenizer, prompt, choice, max_length):
    prompt_ids = tokenizer(prompt, add_special_tokens=False).input_ids
    choice_ids = tokenizer(choice, add_special_tokens=False).input_ids
    input_ids = prompt_ids + choice_ids
    label_mask = [False] * len(prompt_ids) + [True] * len(choice_ids)
    if len(input_ids) > max_length:
        overflow = len(input_ids) - max_length
        input_ids = input_ids[overflow:]
        label_mask = label_mask[overflow:]
    return input_ids, label_mask

def score_choice_batch(model, tokenizer, encoded_choices, max_length):
    pad_id = tokenizer.pad_token_id
    device = next(model.parameters()).device
    max_len = min(max(len(ids) for ids, _ in encoded_choices), max_length)
    input_ids = torch.full((len(encoded_choices), max_len), pad_id, dtype=torch.long)
    attention_mask = torch.zeros((len(encoded_choices), max_len), dtype=torch.long)
    labels = torch.full((len(encoded_choices), max_len), -100, dtype=torch.long)
    for row, (ids, label_mask) in enumerate(encoded_choices):
        ids = ids[-max_len:]
        label_mask = label_mask[-max_len:]
        seq_len = len(ids)
        input_ids[row, :seq_len] = torch.tensor(ids, dtype=torch.long)
        attention_mask[row, :seq_len] = 1
        for col, keep in enumerate(label_mask):
            if keep:
                labels[row, col] = ids[col]
    input_ids = input_ids.to(device)
    attention_mask = attention_mask.to(device)
    labels = labels.to(device)
    with torch.no_grad():
        outputs = model(input_ids=input_ids, attention_mask=attention_mask)
        logits = outputs.logits[:, :-1, :].contiguous()
        shifted_labels = labels[:, 1:].contiguous()
        valid = shifted_labels != -100
        token_losses = F.cross_entropy(
            logits.view(-1, logits.size(-1)),
            shifted_labels.view(-1),
            reduction="none",
            ignore_index=-100,
        ).view(shifted_labels.shape)
    loss_sum = (token_losses * valid).sum(dim=1)
    token_count = valid.sum(dim=1).clamp_min(1)
    scores = (loss_sum / token_count).detach().cpu().tolist()
    del outputs, logits, shifted_labels, valid, token_losses
    del input_ids, attention_mask, labels, loss_sum, token_count
    return scores

def evaluate_task(model, tokenizer, model_name, task_name, examples, batch_size, max_length):
    start_time = time.time()
    correct = 0
    total = 0
    pending = []
    example_choice_counts = []
    example_labels = []
    print("\n" + "-" * 80)
    print(f"Evaluating {model_name} on {task_name}, examples={len(examples)}")
    print("-" * 80)
    def flush_batch():
        nonlocal correct, total, pending, example_choice_counts, example_labels
        if not pending:
            return
        scores = score_choice_batch(model, tokenizer, pending, max_length=max_length)
        offset = 0
        for choice_count, label in zip(example_choice_counts, example_labels):
            cur_scores = scores[offset : offset + choice_count]
            pred = min(range(choice_count), key=lambda idx: cur_scores[idx])
            correct += int(pred == label)
            total += 1
            offset += choice_count
        pending = []
        example_choice_counts = []
        example_labels = []
    for example in tqdm(examples, desc=f"{model_name}/{task_name}"):
        encoded = [
            encode_choice(tokenizer, example.prompt, choice, max_length=max_length)
            for choice in example.choices
        ]
        if len(pending) + len(encoded) > batch_size:
            flush_batch()
        pending.extend(encoded)
        example_choice_counts.append(len(encoded))
        example_labels.append(example.label)
    flush_batch()
    elapsed = time.time() - start_time
    accuracy = correct / max(total, 1)
    print(
        f"{model_name}/{task_name}: accuracy={accuracy:.6%}, "
        f"correct={correct}, total={total}, elapsed={format_time(elapsed)}"
    )
    return TaskResult(
        model_name=model_name,
        model_path=getattr(model, "_zero_shot_model_path", ""),
        task_name=task_name,
        accuracy=accuracy,
        correct=correct,
        total=total,
        elapsed_sec=elapsed,
    )

def release_model(model, tokenizer):
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

def parse_name_path_spec(spec, option_name):
    if "=" not in spec:
        raise ValueError(f"{option_name} must use name=path format, got: {spec}")
    name, path = spec.split("=", 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise ValueError(f"{option_name} must use non-empty name=path format, got: {spec}")
    return name, path

def parse_model_specs(args):
    if args.models_json:
        raw_specs = json.loads(args.models_json)
        return [(str(item["label"]), str(item["path"])) for item in raw_specs]
    if args.model:
        return [parse_name_path_spec(item, "--model") for item in args.model]
    return list(MODEL_SPECS)

def ensure_unique_model_names(model_specs):
    seen = set()
    for model_name, _ in model_specs:
        if model_name in seen:
            raise ValueError(f"Duplicate model name: {model_name}")
        seen.add(model_name)

def validate_model_paths(model_specs, skip_missing_models):
    missing = [(name, path) for name, path in model_specs if not os.path.exists(path)]
    if missing and not skip_missing_models:
        formatted = "\n".join(f"- {name}: {path}" for name, path in missing)
        raise FileNotFoundError(
            "Some configured model paths do not exist. "
            "Fix the path, pass --model name=path, or pass --skip_missing_models.\n"
            f"{formatted}"
        )
    if skip_missing_models:
        return [(name, path) for name, path in model_specs if os.path.exists(path)]
    return model_specs

def build_summary(results):
    by_model = {}
    by_task = {}
    for result in results:
        by_model.setdefault(result.model_name, []).append(result)
        by_task.setdefault(result.task_name, []).append(result)
    model_average_accuracy = {
        model_name: sum(item.accuracy for item in items) / len(items)
        for model_name, items in by_model.items()
    }
    task_rankings = {}
    for task_name, items in by_task.items():
        task_rankings[task_name] = [
            {
                "model_name": item.model_name,
                "accuracy": item.accuracy,
                "correct": item.correct,
                "total": item.total,
            }
            for item in sorted(items, key=lambda value: value.accuracy, reverse=True)
        ]
    return {
        "model_average_accuracy": model_average_accuracy,
        "task_rankings_by_accuracy": task_rankings,
        "interpretation": (
            "Zero-shot accuracy is computed by scoring each answer choice with the "
            "average negative log-likelihood of only the choice tokens. Higher "
            "accuracy is better. The default tasks are PIQA, HellaSwag, and "
            "Winogrande validation splits loaded from the local HuggingFace cache."
        ),
    }

def print_summary(results, output_json=None, model_specs=None, task_names=None, args=None):
    print("\n" + "=" * 80)
    print("Zero-Shot Accuracy Summary")
    print("=" * 80)
    header = f"{'model':38s} {'task':12s} {'accuracy':>12s} {'correct':>10s} {'total':>10s}"
    print(header)
    print("-" * len(header))
    for result in results:
        print(
            f"{result.model_name:38s} {result.task_name:12s} "
            f"{result.accuracy:12.6%} {result.correct:10d} {result.total:10d}"
        )
    print("\nAverage accuracy:")
    summary = build_summary(results)
    for model_name, accuracy in summary["model_average_accuracy"].items():
        print(f"{model_name:38s} avg_acc={accuracy:.6%}")
    if output_json:
        output_dir = os.path.dirname(output_json)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
        payload = {
            "models": [
                {"model_name": model_name, "model_path": model_path}
                for model_name, model_path in (model_specs or [])
            ],
            "tasks": task_names or [],
            "limit": None if args is None else args.limit,
            "batch_size": None if args is None else args.batch_size,
            "max_length": None if args is None else args.max_length,
            "results": [
                {
                    "model_name": result.model_name,
                    "model_path": result.model_path,
                    "task_name": result.task_name,
                    "accuracy": result.accuracy,
                    "correct": result.correct,
                    "total": result.total,
                    "elapsed_sec": result.elapsed_sec,
                }
                for result in results
            ],
            "summary": summary,
        }
        with open(output_json, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2, ensure_ascii=False)
        print(f"\nSaved machine-readable results to: {output_json}")

def print_config(model_specs, task_names, args):
    print("=" * 80)
    print("Configured zero-shot evaluation")
    print("=" * 80)
    print(f"tasks: {', '.join(task_names)}")
    print(f"limit per task: {args.limit}")
    print(f"batch_size: {args.batch_size}")
    print(f"max_length: {args.max_length}")
    print(f"output_json: {args.output_json}")
    print("\nModels:")
    for idx, (model_name, model_path) in enumerate(model_specs, start=1):
        print(f"{idx}. {model_name}: {model_path}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model",
        action="append",
        default=None,
        help=(
            "Model to evaluate in name=path format. Repeat to override the default "
            "six-stage nomark model list."
        ),
    )
    parser.add_argument(
        "--models_json",
        default=None,
        help=(
            "Optional JSON list like "
            "'[{\"label\":\"original\",\"path\":\"/path/to/model\"}]'. "
            "Takes precedence over --model."
        ),
    )
    parser.add_argument(
        "--skip_missing_models",
        action="store_true",
        help="Skip missing configured model paths instead of failing fast.",
    )
    parser.add_argument(
        "--tasks",
        default="piqa,hellaswag,winogrande",
        help="Comma-separated tasks: piqa,hellaswag,winogrande",
    )
    parser.add_argument("--limit", type=int, default=None, help="Optional sample limit per task.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--max_length", type=int, default=512)
    parser.add_argument(
        "--output_json",
        default=DEFAULT_OUTPUT_JSON,
    )
    return parser.parse_args()

def main():
    args = parse_args()
    setup_environment()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluating OPT-125M in this script.")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")
    task_names = [task.strip() for task in args.tasks.split(",") if task.strip()]
    model_specs = parse_model_specs(args)
    ensure_unique_model_names(model_specs)
    model_specs = validate_model_paths(model_specs, args.skip_missing_models)
    if not model_specs:
        raise ValueError("No model paths are available for evaluation.")
    print_config(model_specs, task_names, args)
    task_examples = {
        task_name: load_task_examples(task_name, limit=args.limit)
        for task_name in task_names
    }
    results = []
    for model_name, model_path in model_specs:
        model, tokenizer = load_model_and_tokenizer(model_name, model_path)
        model._zero_shot_model_path = model_path
        for task_name in task_names:
            results.append(
                evaluate_task(
                    model=model,
                    tokenizer=tokenizer,
                    model_name=model_name,
                    task_name=task_name,
                    examples=task_examples[task_name],
                    batch_size=args.batch_size,
                    max_length=args.max_length,
                )
            )
        release_model(model, tokenizer)
    print_summary(
        results,
        output_json=args.output_json,
        model_specs=model_specs,
        task_names=task_names,
        args=args,
    )

if __name__ == "__main__":
    main()
