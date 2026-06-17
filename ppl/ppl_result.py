# evaluate_four_opt13b_models.py
import argparse
import gc
import hashlib
import math
import os
import time
from dataclasses import dataclass

import torch
from datasets import DownloadConfig, load_dataset, load_from_disk
from torch.utils.data import DataLoader, Dataset
from tqdm.auto import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, default_data_collator


BIOLOGY_DATA_PATH = "/root/autodl-tmp/datasets/processed_math1_35k"
DATASET_CACHE_DIR = "/root/autodl-tmp/datasets"

# MODEL_SPECS = [
#     (
#         "original",
#         "/root/autodl-tmp/models/facebook/opt-125m",
#     ),
#     (
#         "original_finetuned_biology",
#         "/root/autodl-tmp/models/opt-125m-finetuned-biology1",
#     ),
#     (
#         "watermarked",
#         "/root/autodl-tmp/models/facebook/opt-1.3b-inserted-by-dssa",
#     ),
#     (
#         "watermarked_finetuned_biology",
#         "/root/autodl-tmp/models/opt-1.3b-inserted-by-dssa-finetuned-biology",
#     ),
# ]

MODEL_SPECS = [
    (
        "original model",
        "/root/autodl-tmp/models/facebook/opt-125m",
    ),
    (
        "watermark model",
        "/root/autodl-tmp/models/facebook/opt-125m-mark1-nomark",
    ),
    (
        "watermaked model finetuned watermarked",
        "/root/autodl-tmp/models/opt-125m-mark1-nomark-ft-math",
    ),
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


def load_biology_validation(data_path):
    return load_biology_split(data_path, "val")


def load_biology_split(data_path, split_name):
    split_path = os.path.join(data_path, split_name)
    if not os.path.exists(split_path):
        raise FileNotFoundError(f"Biology split does not exist: {split_path}")
    dataset = load_from_disk(split_path)
    print("=" * 80)
    print(f"Loaded Biology {split_name} dataset")
    print("=" * 80)
    print(f"path: {split_path}")
    print(f"samples: {len(dataset)}")
    print(f"columns: {dataset.column_names}")
    report_dataset_label_stats(dataset, f"biology/{split_name}")
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
        ignored_label_tokens += len(labels) - valid_labels

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
    print("Biology train/eval exact-overlap check")
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

    pairs = [
        ("original", "original_finetuned_biology"),
        ("watermarked", "watermarked_finetuned_biology"),
    ]

    print("\nBiology learning check:")
    for before_name, after_name in pairs:
        try:
            before = result_lookup(results, before_name, "biology")
            after = result_lookup(results, after_name, "biology")
        except KeyError:
            continue
        ppl_ratio = after.ppl / before.ppl
        loss_drop = before.eval_loss - after.eval_loss
        acc_delta = None
        if before.token_accuracy is not None and after.token_accuracy is not None:
            acc_delta = after.token_accuracy - before.token_accuracy

        print(
            f"{before_name} -> {after_name}: "
            f"biology loss drop={loss_drop:.6f}, "
            f"ppl ratio={ppl_ratio:.6f}, "
            f"token_acc delta={acc_delta:.4%}"
        )

    print("\nGeneral ability retention check on WikiText-2:")
    for before_name, after_name in pairs:
        try:
            before = result_lookup(results, before_name, "wikitext2")
            after = result_lookup(results, after_name, "wikitext2")
        except KeyError:
            continue
        ppl_delta = after.ppl - before.ppl
        ppl_ratio = after.ppl / before.ppl
        print(
            f"{before_name} -> {after_name}: "
            f"wikitext ppl before={before.ppl:.6f}, "
            f"after={after.ppl:.6f}, "
            f"delta={ppl_delta:.6f}, "
            f"ratio={ppl_ratio:.6f}"
        )

    try:
        original = result_lookup(results, "original", "wikitext2")
        watermarked = result_lookup(results, "watermarked", "wikitext2")
        print(
            "\nWatermark insertion base PPL impact: "
            f"original={original.ppl:.6f}, "
            f"watermarked={watermarked.ppl:.6f}, "
            f"delta={watermarked.ppl - original.ppl:.6f}, "
            f"ratio={watermarked.ppl / original.ppl:.6f}"
        )
    except KeyError:
        pass


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--biology_data_path", default=BIOLOGY_DATA_PATH)
    parser.add_argument(
        "--biology_split",
        default="val",
        choices=["train", "val", "test"],
        help="Biology split to evaluate. Default is val; avoid train for generalization PPL.",
    )
    parser.add_argument("--block_size", type=int, default=512)
    parser.add_argument("--biology_batch_size", type=int, default=2)
    parser.add_argument("--wikitext_batch_size", type=int, default=2)
    parser.add_argument(
        "--max_wikitext_blocks",
        type=int,
        default=None,
        help="Optional limit for quick debugging. Default evaluates all WikiText-2 blocks.",
    )
    parser.add_argument(
        "--max_overlap_hashes",
        type=int,
        default=None,
        help="Optional cap for train/eval exact-overlap hashing. Default checks all samples.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    setup_environment()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for evaluating OPT-1.3B in this script.")

    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU count: {torch.cuda.device_count()}")
    print(f"bf16 supported: {torch.cuda.is_bf16_supported()}")

    if args.biology_split == "train":
        print(
            "WARNING: evaluating biology PPL on the training split. "
            "This measures memorization/training fit, not generalization."
        )

    report_train_eval_overlap(
        args.biology_data_path,
        eval_split=args.biology_split,
        max_hashes=args.max_overlap_hashes,
    )
    biology_dataset = load_biology_split(args.biology_data_path, args.biology_split)
    results = []

    for model_name, model_path in MODEL_SPECS:
        model, tokenizer = load_model_and_tokenizer(model_name, model_path)

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
                dataset_name="wikitext2",
                batch_size=args.wikitext_batch_size,
                show_accuracy=False,
            )
        )
        del wikitext_dataset
        gc.collect()

        results.append(
            evaluate_dataset(
                model=model,
                dataset=biology_dataset,
                model_name=model_name,
                model_path=model_path,
                dataset_name="biology",
                batch_size=args.biology_batch_size,
                show_accuracy=True,
            )
        )

        release_model(model, tokenizer)

    print_result_table(results)
    print_conclusions(results)


if __name__ == "__main__":
    main()
