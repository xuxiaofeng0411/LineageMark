
import argparse
import gc
import hashlib
import json
import math
import os
import time
from dataclasses import asdict, dataclass
import torch
from datasets import DownloadConfig, load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator

DOMAIN_DATA_PATH = "/root/autodl-tmp/datasets/processed_law1_35k"
DATASET_CACHE_DIR = "/root/autodl-tmp/datasets"
DEFAULT_OUTPUT_JSON = "/root/autodl-tmp/LineageMark/outputs/ppl_diff_space_law1.json"

MODEL_SPECS = [
    (
        "opt-125m",
        "/root/autodl-tmp/models/facebook/opt-125m",
    ),
    (
        "opt-125m-mark1-full",
        "/root/autodl-tmp/models/facebook/opt-125m-mark1-full",
    ),
    (
        "opt-125m-mark1-fisher",
        "/root/autodl-tmp/models/facebook/opt-125m-mark1-fisher",
    ),
    (
        "opt-125m-mark1-ca",
        "/root/autodl-tmp/models/facebook/opt-125m-mark1-ca",
    ),
    (
        "opt-125m-mark1-full-ft",
        "/root/autodl-tmp/models/facebook/opt-125m-mark1-full-ft",
    ),
    (
        "opt-125m-mark1-fisher-ft",
        "/root/autodl-tmp/models/facebook/opt-125m-mark1-fisher-ft",
    ),
    (
        "opt-125m-mark1-ca-ft",
        "/root/autodl-tmp/models/facebook/opt-125m-mark1-ca-ft",
    ),
    # (
    #     "dssa-s42-r005-m010-wm-ft",
    #     "/root/autodl-tmp/models/facebook/DSSA-space2/dssa-s42-r005-m010-wm-ft",
    # ),
    # (
    #     "dssa-s42-r005-m020-wm-ft",
    #     "/root/autodl-tmp/models/facebook/DSSA-space2/dssa-s42-r005-m020-wm-ft",
    # ),
    # (
    #     "dssa-s42-r010-m005-wm-ft",
    #     "/root/autodl-tmp/models/facebook/DSSA-space2/dssa-s42-r010-m005-wm-ft",
    # ),
    # (
    #     "dssa-s42-r010-m010-wm-ft",
    #     "/root/autodl-tmp/models/facebook/DSSA-space2/dssa-s42-r010-m010-wm-ft",
    # ),
    # (
    #     "dssa-s42-r010-m020-wm-ft",
    #     "/root/autodl-tmp/models/facebook/DSSA-space2/dssa-s42-r010-m020-wm-ft",
    # ),
    # (
    #     "dssa-s42-r025-m005-wm-ft",
    #     "/root/autodl-tmp/models/facebook/DSSA-space2/dssa-s42-r025-m005-wm-ft",
    # ),
    # (
    #     "dssa-s42-r025-m010-wm-ft",
    #     "/root/autodl-tmp/models/facebook/DSSA-space2/dssa-s42-r025-m010-wm-ft",
    # ),
    # (
    #     "dssa-s42-r025-m020-wm-ft",
    #     "/root/autodl-tmp/models/facebook/DSSA-space2/dssa-s42-r025-m020-wm-ft",
    # ),
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
        if input_ids.ndim != 1:
            raise ValueError("input_ids must be a 1D tensor")
        usable_length = (input_ids.numel() // block_size) * block_size
        if usable_length == 0:
            raise ValueError("not enough tokens to build evaluation blocks")
        self.input_ids = input_ids[:usable_length]
        self.block_size = block_size

    def __len__(self):
        return self.input_ids.numel() // self.block_size

    def __getitem__(self, idx):
        start = idx * self.block_size
        end = start + self.block_size
        ids = self.input_ids[start:end]
        return {
            "input_ids": ids,
            "attention_mask": torch.ones_like(ids),
            "labels": ids.clone(),
        }

def setup_environment():
    os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
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

def safe_ppl(loss):
    return math.exp(loss) if loss < 100 else float("inf")

def load_domain_validation(data_path):
    return load_domain_split(data_path, "val")

def load_domain_split(data_path, split_name):
    split_path = os.path.join(data_path, split_name)
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Domain split does not exist: {split_path}")
    dataset = load_from_disk(split_path)
    print("=" * 80)
    print(f"Loaded domain {split_name} dataset")
    print("=" * 80)
    print(f"path: {split_path}")
    print(f"samples: {len(dataset)}")
    print(f"columns: {dataset.column_names}")
    report_dataset_label_stats(dataset, f"domain/{split_name}")
    return dataset

def _ids_to_list(value):
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().tolist()
    return list(value)

def _hash_ids(value):
    ids = _ids_to_list(value)
    payload = ",".join(str(int(x)) for x in ids)
    return hashlib.md5(payload.encode()).hexdigest()

def report_dataset_label_stats(dataset, dataset_name, max_samples=256):
    if len(dataset) == 0:
        print(f"{dataset_name}: empty dataset")
        return
    sample_count = min(max_samples, len(dataset))
    total_input_tokens = 0
    total_attention_tokens = 0
    total_label_tokens = 0
    all_label_tokens = 0
    ignored_label_tokens = 0
    for idx in range(sample_count):
        item = dataset[idx]
        input_ids = _ids_to_list(item["input_ids"])
        attention_mask = _ids_to_list(item.get("attention_mask", [1] * len(input_ids)))
        labels = _ids_to_list(item.get("labels", input_ids))
        total_input_tokens += len(input_ids)
        total_attention_tokens += sum(int(x) for x in attention_mask)
        valid_labels = sum(1 for x in labels if int(x) != -100)
        total_label_tokens += valid_labels
        all_label_tokens += len(labels)
        ignored_label_tokens += len(labels) - valid_label
    print(
        f"{dataset_name} label stats over {sample_count} samples: "
        f"avg_input_tokens={total_input_tokens / sample_count:.2f}, "
        f"avg_attention_tokens={total_attention_tokens / sample_count:.2f}, "
        f"avg_valid_label_tokens={total_label_tokens / sample_count:.2f}, "
        f"ignored_label_ratio={ignored_label_tokens / max(all_label_tokens, 1):.4%}"
    )

def report_train_eval_overlap(data_path, eval_split="val", max_hashes=None):
    train_path = os.path.join(data_path, "train")
    eval_path = os.path.join(data_path, eval_split)
    if not os.path.exists(train_path) or not os.path.exists(eval_path):
        print("Train/eval overlap check skipped: missing train or eval split.")
        return
    train_dataset = load_from_disk(train_path)
    eval_dataset = load_from_disk(eval_path)
    train_limit = len(train_dataset) if max_hashes is None else min(max_hashes, len(train_dataset))
    eval_limit = len(eval_dataset) if max_hashes is None else min(max_hashes, len(eval_dataset))
    train_hashes = set()
    for idx in range(train_limit):
        train_hashes.add(_hash_ids(train_dataset[idx]["input_ids"]))
    overlap = 0
    for idx in range(eval_limit):
        if _hash_ids(eval_dataset[idx]["input_ids"]) in train_hashes:
            overlap += 1
    print("=" * 80)
    print("Domain train/eval exact-overlap check")
    print("=" * 80)
    print(f"train samples checked: {train_limit}")
    print(f"{eval_split} samples checked: {eval_limit}")
    print(f"exact input_id overlap: {overlap}")
    print(f"overlap ratio in {eval_split}: {overlap / max(eval_limit, 1):.4%}")

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
        kwargs = {
            "path": path,
            "name": "wikitext-2-raw-v1",
            "split": "test",
        }
        if cache_dir is not None:
            kwargs["cache_dir"] = cache_dir
        if local_only:
            kwargs["download_config"] = DownloadConfig(local_files_only=True)
        try:
            raw = load_dataset(**kwargs)
            print(
                "Loaded WikiText-2 with "
                f"path={path}, cache_dir={cache_dir or 'default'}, "
                f"local_only={local_only}"
            )
            return raw
        except Exception as exc:
            errors.append(
                f"path={path}, cache_dir={cache_dir or 'default'}, "
                f"local_only={local_only}: {type(exc).__name__}: {exc}"
            )
    joined = "\n".join(errors)
    raise RuntimeError(f"Could not load WikiText-2 test split. Attempts:\n{joined}")

def load_general_wikitext_dataset(tokenizer, block_size, max_blocks=None):
    print("=" * 80)
    print("Loading WikiText-2 test dataset")
    print("=" * 80)
    raw = load_wikitext_raw_split()
    text = "\n\n".join(t for t in raw["text"] if t and not t.isspace())
    encoded = tokenizer(text, return_tensors="pt")
    input_ids = encoded.input_ids[0]
    dataset = TokenizedTextDataset(input_ids, block_size=block_size)
    if max_blocks is not None:
        max_len = min(max_blocks, len(dataset))
        dataset.input_ids = dataset.input_ids[: max_len * block_size]
    print(f"raw samples: {len(raw)}")
    print(f"tokens: {dataset.input_ids.numel()}")
    print(f"blocks: {len(dataset)}")
    print(f"block_size: {block_size}")
    return dataset

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

def evaluate_dataset(model, dataset, model_name, model_path, dataset_name, batch_size, show_accuracy):
    device = next(model.parameters()).device
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=default_data_collator,
    )
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
            labels = batch["labels"].to(device)
            labels = labels.masked_fill(attention_mask == 0, -100)
            outputs = model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
            )
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
    ppl = safe_ppl(avg_loss)
    token_accuracy = None
    if show_accuracy:
        token_accuracy = correct_tokens / max(total_accuracy_tokens, 1)
    print(f"eval_loss: {avg_loss:.6f}")
    print(f"ppl: {ppl:.6f}")
    if token_accuracy is not None:
        print(f"token_accuracy: {token_accuracy:.6%}")
    print(f"loss_tokens: {total_loss_tokens}")
    print(f"elapsed: {format_time(elapsed)}")
    return EvalResult(
        model_name=model_name,
        model_path=model_path,
        dataset_name=dataset_name,
        eval_loss=avg_loss,
        ppl=ppl,
        token_accuracy=token_accuracy,
        loss_tokens=total_loss_tokens,
        elapsed_sec=elapsed,
    )

def release_model(model, tokenizer):
    del model, tokenizer
    gc.collect()
    torch.cuda.empty_cache()

def print_result_table(results):
    print("\n" + "=" * 80)
    print("Evaluation Summary")
    print("=" * 80)
    header = (
        f"{'model':38s} {'dataset':14s} {'loss':>10s} {'ppl':>12s} "
        f"{'token_acc':>12s} {'tokens':>12s}"
    )
    print(header)
    print("-" * len(header))
    for result in results:
        acc = "-" if result.token_accuracy is None else f"{result.token_accuracy:.4%}"
        print(
            f"{result.model_name:38s} {result.dataset_name:14s} "
            f"{result.eval_loss:10.6f} {result.ppl:12.6f} "
            f"{acc:>12s} {result.loss_tokens:12d}"
        )

def result_lookup(results, model_name, dataset_name):
    for result in results:
        if result.model_name == model_name and result.dataset_name == dataset_name:
            return result
    raise KeyError((model_name, dataset_name))

def print_conclusions(results):
    print("\n" + "=" * 80)
    print("Comparisons")
    print("=" * 80)
    clean_by_dataset = {
        result.dataset_name: result
        for result in results
        if result.model_name == "clean_ft"
    }
    if not clean_by_dataset:
        print("No clean_ft result found; skip delta-PPL comparisons.")
        return
    for result in results:
        clean = clean_by_dataset.get(result.dataset_name)
        if clean is None or result.model_name == "clean_ft":
            continue
        ppl_delta = result.ppl - clean.ppl
        ppl_ratio = result.ppl / clean.ppl
        loss_delta = result.eval_loss - clean.eval_loss
        print(
            f"{result.model_name} vs clean_ft on {result.dataset_name}: "
            f"delta_loss={loss_delta:.6f}, "
            f"delta_ppl={ppl_delta:.6f}, "
            f"ppl_ratio={ppl_ratio:.6f}"
        )

def parse_name_path_spec(spec, option_name):
    if '=' not in spec:
        raise ValueError(f'{option_name} must use name=path format, got: {spec}')
    name, path = spec.split('=', 1)
    name = name.strip()
    path = path.strip()
    if not name or not path:
        raise ValueError(f'{option_name} must use non-empty name=path format, got: {spec}')
    return name, path

def parse_model_specs(args):
    if args.models_json:
        raw_specs = json.loads(args.models_json)
        return [(str(item['label']), str(item['path'])) for item in raw_specs]
    if args.model:
        return [parse_name_path_spec(item, '--model') for item in args.model]
    return list(MODEL_SPECS)

def parse_dataset_specs(args):
    dataset_specs = []
    if args.dataset:
        dataset_specs.extend(parse_name_path_spec(item, '--dataset') for item in args.dataset)
    if args.dataset_path:
        dataset_specs.append((args.dataset_name or 'domain', args.dataset_path))
    if not dataset_specs:
        dataset_specs.append((args.dataset_name or 'pubmed1', DOMAIN_DATA_PATH))
    return dataset_specs

def validate_model_specs(model_specs, skip_missing_models):
    missing = [(name, path) for name, path in model_specs if not os.path.exists(path)]
    if missing and not skip_missing_models:
        formatted = '\n'.join(f'- {name}: {path}' for name, path in missing)
        raise FileNotFoundError('Some model paths do not exist:\n' + formatted)
    if skip_missing_models:
        return [(name, path) for name, path in model_specs if os.path.exists(path)]
    return model_specs

def validate_dataset_specs(dataset_specs, split_name):
    missing = []
    for name, path in dataset_specs:
        split_path = os.path.join(path, split_name)
        if not os.path.exists(split_path):
            missing.append(f'- {name}: {split_path}')
    if missing:
        raise FileNotFoundError('Some dataset split paths do not exist:\n' + '\n'.join(missing))

def save_results(results, model_specs, dataset_specs, args):
    if not args.output_json:
        return
    payload = {
        'models': [{'model_name': name, 'model_path': path} for name, path in model_specs],
        'datasets': [{'dataset_name': name, 'dataset_path': path, 'split': args.split} for name, path in dataset_specs],
        'include_wikitext': not args.skip_wikitext,
        'block_size': args.block_size,
        'results': [asdict(result) for result in results],
    }
    output_dir = os.path.dirname(args.output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)
    print(f'Saved machine-readable PPL results to: {args.output_json}')

def parse_args():
    parser = argparse.ArgumentParser(description='Evaluate model PPL on WikiText-2 and one or more local domain datasets.')
    parser.add_argument('--model', action='append', default=None, help='Model in label=path format. Repeat to evaluate multiple models.')
    parser.add_argument('--models_json', default=None, help='JSON list of objects with label and path keys. Takes precedence over --model.')
    parser.add_argument('--dataset', action='append', default=None, help='Domain dataset in label=path format. Repeat to evaluate multiple datasets. Path must contain the selected split directory.')
    parser.add_argument('--dataset_path', '--data_path', dest='dataset_path', default=None, help='Single domain dataset path. Equivalent to --dataset dataset_name=path.')
    parser.add_argument('--dataset_name', default=None, help='Label used with --dataset_path. Default: domain, or pubmed1 for the built-in default path.')
    parser.add_argument('--split', dest='split', default='val', choices=['train', 'val', 'test'], help='Dataset split to evaluate.')
    parser.add_argument('--block_size', type=int, default=512)
    parser.add_argument('--domain_batch_size', dest='domain_batch_size', type=int, default=2)
    parser.add_argument('--wikitext_batch_size', type=int, default=2)
    parser.add_argument('--max_wikitext_blocks', type=int, default=None, help='Optional limit for quick debugging. Default evaluates all WikiText-2 blocks.')
    parser.add_argument('--max_overlap_hashes', type=int, default=None, help='Optional cap for train/eval exact-overlap hashing. Default checks all samples.')
    parser.add_argument('--skip_wikitext', action='store_true', help='Only evaluate domain datasets, not WikiText-2.')
    parser.add_argument('--skip_overlap_check', action='store_true', help='Skip train/eval exact-overlap hashing for local domain datasets.')
    parser.add_argument('--skip_missing_models', action='store_true', help='Skip missing model paths instead of failing.')
    parser.add_argument('--output_json', default=DEFAULT_OUTPUT_JSON, help='Optional JSON output path.')
    return parser.parse_args()

def main():
    args = parse_args()
    setup_environment()
    if not torch.cuda.is_available():
        raise RuntimeError('CUDA is required for PPL evaluation.')
    print(f'GPU: {torch.cuda.get_device_name(0)}')
    print(f'GPU count: {torch.cuda.device_count()}')
    print(f'bf16 supported: {torch.cuda.is_bf16_supported()}')
    model_specs = validate_model_specs(parse_model_specs(args), args.skip_missing_models)
    if not model_specs:
        raise ValueError('No model paths are available for evaluation.')
    dataset_specs = parse_dataset_specs(args)
    validate_dataset_specs(dataset_specs, args.split)
    if args.split == 'train':
        print('WARNING: evaluating PPL on the training split. This measures training fit, not generalization.')
    if not args.skip_overlap_check:
        for dataset_name, dataset_path in dataset_specs:
            print(f'Checking train/{args.split} exact-overlap for {dataset_name}')
            report_train_eval_overlap(dataset_path, eval_split=args.split, max_hashes=args.max_overlap_hashes)
    domain_datasets = [
        (dataset_name, load_domain_split(dataset_path, args.split))
        for dataset_name, dataset_path in dataset_specs
    ]
    results = []
    for model_name, model_path in model_specs:
        model, tokenizer = load_model_and_tokenizer(model_name, model_path)
        if not args.skip_wikitext:
            wikitext_dataset = load_general_wikitext_dataset(
                tokenizer=tokenizer,
                block_size=args.block_size,
                max_blocks=args.max_wikitext_blocks,
            )
            results.append(
                evaluate_dataset(
                    model=model,
                    dataset=wikitext_dataset,
                    model_name=model_name,
                    model_path=model_path,
                    dataset_name='wikitext2',
                    batch_size=args.wikitext_batch_size,
                    show_accuracy=False,
                )
            )
            del wikitext_dataset
            gc.collect()
        for dataset_name, dataset in domain_datasets:
            results.append(
                evaluate_dataset(
                    model=model,
                    dataset=dataset,
                    model_name=model_name,
                    model_path=model_path,
                    dataset_name=dataset_name,
                    batch_size=args.domain_batch_size,
                    show_accuracy=True,
                )
            )
        release_model(model, tokenizer)
    print_result_table(results)
    print_conclusions(results)
    save_results(results, model_specs, dataset_specs, args)
if __name__ == "__main__":
    main()
