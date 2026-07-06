import argparse
import gc
import json
import math
import os
import time
from dataclasses import asdict, dataclass
from pathlib import Path
import torch
from datasets import DownloadConfig, load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator

DOMAIN = "pubmed"
DATASET_CACHE_DIR = "/root/autodl-tmp/datasets"
OUTPUT_JSON = "/root/autodl-tmp/project12/outputs/ppl_result_pubmed.json"

MODEL_SPECS = [
    ("original", "/root/autodl-tmp/models/facebook/opt-125m"),
    ("pubmed_ft_stage1", "/root/autodl-tmp/models/pubmed/opt-125m-mark1-nomark-ft-pubmed"),
    ("pubmed_ft_stage2", "/root/autodl-tmp/models/pubmed/opt-125m-mark2-nomark-ft-pubmed"),
    ("pubmed_ft_stage3", "/root/autodl-tmp/models/pubmed/opt-125m-mark3-nomark-ft-pubmed"),
    ("pubmed_watermark_stage1", "/root/autodl-tmp/models/facebook/pubmed/opt-125m-mark1-nomark"),
    ("pubmed_watermark_stage2", "/root/autodl-tmp/models/facebook/pubmed/opt-125m-mark2-nomark"),
    ("pubmed_watermark_stage3", "/root/autodl-tmp/models/facebook/pubmed/opt-125m-mark3-nomark"),
    ("pubmed_watermark_stage4", "/root/autodl-tmp/models/facebook/pubmed/opt-125m-mark4-nomark"),
]
DOMAIN_DATASETS = [
    ("pubmed_stage1", "/root/autodl-tmp/datasets/processed_pubmed1_35k"),
    ("pubmed_stage2", "/root/autodl-tmp/datasets/processed_pubmed2_35k"),
    ("pubmed_stage3", "/root/autodl-tmp/datasets/processed_pubmed3_35k"),
]

@dataclass
class EvalResult:
    model_name: str
    model_path: str
    dataset_name: str
    eval_loss: float
    ppl: float
    token_accuracy: float | None
    loss_tokens: int
    elapsed_sec: float

class TokenizedTextDataset(Dataset):
    def __init__(self, input_ids, block_size):
        usable_length = (input_ids.numel() // block_size) * block_size
        if usable_length == 0:
            raise ValueError("not enough tokens to build evaluation blocks")
        self.input_ids = input_ids[:usable_length]
        self.block_size = block_size

    def __len__(self):
        return self.input_ids.numel() // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        ids = self.input_ids[start : start + self.block_size]
        return {"input_ids": ids, "attention_mask": torch.ones_like(ids), "labels": ids.clone()}

def setup_environment():
    os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    torch.backends.cuda.matmul.allow_tf32 = True

def safe_ppl(loss):
    return math.exp(loss) if loss < 100 else float("inf")

def format_time(seconds):
    minutes, sec = divmod(int(seconds), 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours}h {minutes}m {sec}s"
    if minutes:
        return f"{minutes}m {sec}s"
    return f"{sec}s"

def require_paths():
    missing = []
    for _, path in MODEL_SPECS:
        if not Path(path).exists():
            missing.append(path)
    for _, path in DOMAIN_DATASETS:
        if not Path(path, "val").exists():
            missing.append(str(Path(path, "val")))
    if missing:
        raise FileNotFoundError("Missing required paths:\n" + "\n".join(missing))

def load_model_and_tokenizer(model_name, model_path):
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
    return model, tokenizer

def load_wikitext_raw_split():
    attempts = [
        ("Salesforce/wikitext", None, True),
        ("wikitext", None, True),
        ("Salesforce/wikitext", DATASET_CACHE_DIR, True),
        ("wikitext", DATASET_CACHE_DIR, True),
        ("Salesforce/wikitext", None, False),
        ("wikitext", None, False),
    ]
    errors = []
    for path, cache_dir, local_only in attempts:
        kwargs = {"path": path, "name": "wikitext-2-raw-v1", "split": "test"}
        if cache_dir is not None:
            kwargs["cache_dir"] = cache_dir
        if local_only:
            kwargs["download_config"] = DownloadConfig(local_files_only=True)
        try:
            raw = load_dataset(**kwargs)
            print(f"Loaded WikiText-2 with path={path}, cache_dir={cache_dir or 'default'}, local_only={local_only}")
            return raw
        except Exception as exc:
            errors.append(f"{path}, cache={cache_dir}, local_only={local_only}: {type(exc).__name__}: {exc}")
    raise RuntimeError("Could not load WikiText-2 test split:\n" + "\n".join(errors))

def load_wikitext_dataset(tokenizer, block_size, max_blocks=None):
    raw = load_wikitext_raw_split()
    text = "\n\n".join(t for t in raw["text"] if t and not t.isspace())
    input_ids = tokenizer(text, return_tensors="pt").input_ids[0]
    dataset = TokenizedTextDataset(input_ids, block_size)
    if max_blocks is not None:
        dataset.input_ids = dataset.input_ids[: min(max_blocks, len(dataset)) * block_size]
    print(f"WikiText blocks: {len(dataset)}, tokens: {dataset.input_ids.numel()}")
    return dataset

def load_domain_dataset(dataset_path, split_name):
    split_path = Path(dataset_path) / split_name
    dataset = load_from_disk(str(split_path))
    print(f"Loaded {split_path}: samples={len(dataset)}, columns={dataset.column_names}")
    return dataset

def evaluate_dataset(model, dataset, model_name, model_path, dataset_name, batch_size, show_accuracy):
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=default_data_collator)
    total_nll = 0.0
    total_loss_tokens = 0
    correct_tokens = 0
    total_accuracy_tokens = 0
    start_time = time.time()
    print("\n" + "-" * 80)
    print(f"Evaluating {model_name} on {dataset_name}")
    print("-" * 80)
    with torch.no_grad():
        for batch in tqdm(loader, desc=f"{model_name}/{dataset_name}"):
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device).masked_fill(attention_mask == 0, -100)
            outputs = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            shift_labels = labels[:, 1:]
            valid_mask = shift_labels != -100
            token_count = int(valid_mask.sum().item())
            total_nll += float(outputs.loss.item()) * token_count
            total_loss_tokens += token_count
            if show_accuracy:
                predictions = outputs.logits[:, :-1, :].argmax(dim=-1)
                correct_tokens += int((predictions.eq(shift_labels) & valid_mask).sum().item())
                total_accuracy_tokens += token_count
                del predictions
            del outputs, input_ids, attention_mask, labels, shift_labels, valid_mask
    elapsed = time.time() - start_time
    avg_loss = total_nll / max(total_loss_tokens, 1)
    token_accuracy = correct_tokens / max(total_accuracy_tokens, 1) if show_accuracy else None
    result = EvalResult(model_name, model_path, dataset_name, avg_loss, safe_ppl(avg_loss), token_accuracy, total_loss_tokens, elapsed)
    print(f"eval_loss: {result.eval_loss:.6f}")
    print(f"ppl: {result.ppl:.6f}")
    if result.token_accuracy is not None:
        print(f"token_accuracy: {result.token_accuracy:.6%}")
    print(f"loss_tokens: {result.loss_tokens}")
    print(f"elapsed: {format_time(result.elapsed_sec)}")
    return result

def release_model(model, tokenizer):
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

def print_result_table(results):
    print("\n" + "=" * 100)
    print(f"{DOMAIN} PPL Evaluation Summary")
    print("=" * 100)
    header = f"{'model':32s} {'dataset':18s} {'loss':>10s} {'ppl':>12s} {'token_acc':>12s} {'tokens':>12s}"
    print(header)
    print("-" * len(header))
    for result in results:
        acc = "-" if result.token_accuracy is None else f"{result.token_accuracy:.4%}"
        print(f"{result.model_name:32s} {result.dataset_name:18s} {result.eval_loss:10.6f} {result.ppl:12.6f} {acc:>12s} {result.loss_tokens:12d}")

def save_results(results, output_json):
    Path(output_json).parent.mkdir(parents=True, exist_ok=True)
    payload = {"domain": DOMAIN, "results": [asdict(result) for result in results]}
    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f"Saved results: {output_json}")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--split", default="val", choices=["train", "val"])
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--domain_batch_size", type=int, default=2)
    parser.add_argument("--wikitext_batch_size", type=int, default=2)
    parser.add_argument("--max_wikitext_blocks", type=int, default=None)
    parser.add_argument("--output_json", default=None)
    return parser.parse_args()

def main():
    args = parse_args()
    setup_environment()
    require_paths()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for PPL evaluation.")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    domain_datasets = [(name, load_domain_dataset(path, args.split)) for name, path in DOMAIN_DATASETS]
    results = []
    for model_name, model_path in MODEL_SPECS:
        model, tokenizer = load_model_and_tokenizer(model_name, model_path)
        wikitext_dataset = load_wikitext_dataset(tokenizer, args.block_size, args.max_wikitext_blocks)
        results.append(evaluate_dataset(model, wikitext_dataset, model_name, model_path, "wikitext2", args.wikitext_batch_size, False))
        del wikitext_dataset
        gc.collect()
        for dataset_name, dataset in domain_datasets:
            results.append(evaluate_dataset(model, dataset, model_name, model_path, dataset_name, args.domain_batch_size, True))
        release_model(model, tokenizer)
    print_result_table(results)
    save_results(results, args.output_json or OUTPUT_JSON)

if __name__ == "__main__":
    main()
