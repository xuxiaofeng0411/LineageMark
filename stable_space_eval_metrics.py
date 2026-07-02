#!/usr/bin/env python3

"""
Evaluate stable-space ablations for LineageMark.
The script compares stable carrier masks produced by different subspace
construction methods after identical watermark embedding and full-parameter
fine-tuning.
Default run for the prepared OPT-125M ablation:
python -u stable_space_eval_metrics.py
The command-line arguments can still override the built-in defaults when running
a different model, watermark, or subspace set.
"""

import argparse
import gc
import hashlib
import json
import math
import os
from dataclasses import dataclass
from types import SimpleNamespace
import torch
from transformers import AutoModelForCausalLM
from lib_multi.utils_multi import (
    find_layers,
    get_blocks,
    get_projection_bit_mapping,
    is_watermark_row,
    qweight2weight,
    string_to_binary,
)

@dataclass
class MethodSpec:
    name: str
    subspace_path: str
    watermarked_model: str
    finetuned_model: str

DEFAULT_PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
DEFAULT_MODEL_ROOT = "/root/autodl-tmp/models/facebook"
DEFAULT_HIDDEN_SIZE = 768
DEFAULT_PASSWORD = "qihdnbji"
DEFAULT_WATERMARK = "bear"
DEFAULT_GAMMA_1 = 2
DEFAULT_XI = 4
DEFAULT_CHUNK_LENGTH = 8
DEFAULT_METHOD_SPECS = [
    MethodSpec(
        "full",
        os.path.join(DEFAULT_MODEL_ROOT, "opt-125m-mark1-subspace-full"),
        os.path.join(DEFAULT_MODEL_ROOT, "opt-125m-mark1-full"),
        os.path.join(DEFAULT_MODEL_ROOT, "opt-125m-mark1-full-ft"),
    ),
    MethodSpec(
        "fisher_only",
        os.path.join(DEFAULT_MODEL_ROOT, "opt-125m-mark1-subspace-fisher"),
        os.path.join(DEFAULT_MODEL_ROOT, "opt-125m-mark1-fisher"),
        os.path.join(DEFAULT_MODEL_ROOT, "opt-125m-mark1-fisher-ft"),
    ),
    MethodSpec(
        "ca_only",
        os.path.join(DEFAULT_MODEL_ROOT, "opt-125m-mark1-subspace-ca"),
        os.path.join(DEFAULT_MODEL_ROOT, "opt-125m-mark1-ca"),
        os.path.join(DEFAULT_MODEL_ROOT, "opt-125m-mark1-ca-ft"),
    ),
]
DEFAULT_UTILITY_JSONS = [
    os.path.join(DEFAULT_PROJECT_ROOT, "outputs", "ppl_diff_space_law1.json")
]
DEFAULT_OUTPUT = os.path.join(DEFAULT_PROJECT_ROOT, "outputs", "stable_space_eval_law1.json")

def parse_method_spec(value):
    parts = value.split(":", 3)
    if len(parts) != 4:
        raise argparse.ArgumentTypeError(
            "--method_spec must be name:subspace_path:watermarked_model:finetuned_model"
        )
    return MethodSpec(*parts)

def load_pt(path):
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        return torch.load(path, map_location="cpu")

def load_masks(path):
    saved = load_pt(path)
    if isinstance(saved, dict) and "all_layer_masks" in saved:
        return saved["all_layer_masks"], saved.get("mapping_config", {})
    if isinstance(saved, dict) and all(isinstance(k, int) for k in saved.keys()):
        return saved, {}
    raise ValueError(f"Cannot find all_layer_masks in {path}")

def load_model(path, device):
    kwargs = {"torch_dtype": "auto", "trust_remote_code": False}
    if device != "cpu":
        kwargs["device_map"] = device
    model = AutoModelForCausalLM.from_pretrained(path, **kwargs)
    model.eval()
    return model

def layer_selected(args, layer_id):
    layer_key = int(hashlib.md5((args.password + str(layer_id)).encode()).hexdigest(), 16)
    return layer_key % args.gamma_1 == 0

def weight_matrix(module):
    if hasattr(module, "qweight"):
        return qweight2weight(module).data.t().detach().cpu().to(torch.float32)
    return module.weight.data.detach().cpu().to(torch.float32)

def bool_mask(mask_info):
    mask = mask_info["W_mask"] if isinstance(mask_info, dict) else mask_info
    if torch.is_tensor(mask):
        return mask.detach().cpu().bool()
    return torch.as_tensor(mask, dtype=torch.bool)

def safe_mean(total, count):
    return float(total / count) if count else None

def metric_ratio(numerator, denominator, eps=1e-12):
    if numerator is None or denominator is None:
        return None
    return float(numerator / (denominator + eps))

def compute_drift_for_scope(wm_model, ft_model, masks, args, selected_layers_only):
    wm_blocks = get_blocks(wm_model)
    ft_blocks = get_blocks(ft_model)
    selected_sum = 0.0
    selected_count = 0
    selected_ref_sum = 0.0
    unselected_sum = 0.0
    unselected_count = 0
    unselected_ref_sum = 0.0
    matrix_rows = []
    layer_id = 0
    for block_idx, (wm_block, ft_block) in enumerate(zip(wm_blocks, ft_blocks)):
        wm_layers = find_layers(wm_block)
        ft_layers = find_layers(ft_block)
        block_masks = masks.get(block_idx, {})
        for name, wm_module in wm_layers.items():
            if name not in ft_layers:
                layer_id += 1
                continue
            is_selected_layer = layer_selected(args, layer_id)
            if selected_layers_only and not is_selected_layer:
                layer_id += 1
                continue
            if name not in block_masks:
                layer_id += 1
                continue
            W_before = weight_matrix(wm_module)
            W_after = weight_matrix(ft_layers[name])
            mask = bool_mask(block_masks[name])
            if mask.shape != W_before.shape or W_after.shape != W_before.shape:
                print(
                    f"Warning: skip block {block_idx} [{name}] shape mismatch: "
                    f"mask={tuple(mask.shape)} before={tuple(W_before.shape)} after={tuple(W_after.shape)}"
                )
                layer_id += 1
                continue
            diff = torch.abs(W_after - W_before)
            ref = torch.abs(W_before)
            inv_mask = ~mask
            sel_count = int(mask.sum().item())
            unsel_count = int(inv_mask.sum().item())
            sel_sum = float(diff[mask].sum().item()) if sel_count else 0.0
            unsel_sum = float(diff[inv_mask].sum().item()) if unsel_count else 0.0
            sel_ref_sum = float(ref[mask].sum().item()) if sel_count else 0.0
            unsel_ref_sum = float(ref[inv_mask].sum().item()) if unsel_count else 0.0
            selected_sum += sel_sum
            selected_count += sel_count
            selected_ref_sum += sel_ref_sum
            unselected_sum += unsel_sum
            unselected_count += unsel_count
            unselected_ref_sum += unsel_ref_sum
            sel_mean = safe_mean(sel_sum, sel_count)
            unsel_mean = safe_mean(unsel_sum, unsel_count)
            matrix_rows.append({
                "block_idx": block_idx,
                "layer_id": layer_id,
                "layer_name": name,
                "watermark_selected_layer": is_selected_layer,
                "selected_count": sel_count,
                "unselected_count": unsel_count,
                "selected_drift": sel_mean,
                "unselected_drift": unsel_mean,
                "drift_ratio": metric_ratio(sel_mean, unsel_mean),
                "selected_relative_drift": metric_ratio(sel_sum, sel_ref_sum),
                "unselected_relative_drift": metric_ratio(unsel_sum, unsel_ref_sum),
            })
            layer_id += 1
    selected_mean = safe_mean(selected_sum, selected_count)
    unselected_mean = safe_mean(unselected_sum, unselected_count)
    return {
        "selected_layers_only": selected_layers_only,
        "selected_count": selected_count,
        "unselected_count": unselected_count,
        "selected_drift": selected_mean,
        "unselected_drift": unselected_mean,
        "drift_ratio": metric_ratio(selected_mean, unselected_mean),
        "selected_relative_drift": metric_ratio(selected_sum, selected_ref_sum),
        "unselected_relative_drift": metric_ratio(unselected_sum, unselected_ref_sum),
        "per_matrix": matrix_rows,
    }

def projection_scope_mask(scope, masks, block_idx, layer_name):
    if scope == "detector":
        return None
    block_masks = masks.get(block_idx, {})
    if layer_name not in block_masks:
        return None
    return bool_mask(block_masks[layer_name])

def compute_projection_margin_for_scope(wm_model, ft_model, masks, args, scope):
    wm_blocks = get_blocks(wm_model)
    ft_blocks = get_blocks(ft_model)
    watermark_bits = string_to_binary(args.watermark)
    if len(watermark_bits) % args.chunk_length != 0:
        raise ValueError("watermark bit length must be divisible by chunk_length")
    chunk_num = len(watermark_bits) // args.chunk_length
    eps = args.margin_eps
    records = []
    before_sum = 0.0
    after_sum = 0.0
    drop_sum = 0.0
    retention_sum = 0.0
    count = 0
    positive_before = 0
    positive_after = 0
    sign_flip = 0
    empty_groups = 0
    layer_id = 0
    selected_matrix_id = 0
    map_args = SimpleNamespace(
        password=args.password,
        hidden_size=args.hidden_size,
        xi=args.xi,
    )
    for block_idx, (wm_block, ft_block) in enumerate(zip(wm_blocks, ft_blocks)):
        wm_layers = find_layers(wm_block)
        ft_layers = find_layers(ft_block)
        for name, wm_module in wm_layers.items():
            if name not in ft_layers:
                layer_id += 1
                continue
            if not layer_selected(args, layer_id):
                layer_id += 1
                continue
            W_before = weight_matrix(wm_module)
            W_after = weight_matrix(ft_layers[name])
            mask = projection_scope_mask(scope, masks, block_idx, name)
            if scope == "mask" and mask is None:
                layer_id += 1
                selected_matrix_id += 1
                continue
            if mask is not None and mask.shape != W_before.shape:
                print(
                    f"Warning: skip projection margin block {block_idx} [{name}] "
                    f"shape mismatch mask={tuple(mask.shape)} weight={tuple(W_before.shape)}"
                )
                layer_id += 1
                selected_matrix_id += 1
                continue
            chunk_id = selected_matrix_id % chunk_num
            projections_before = [0.0] * args.chunk_length
            projections_after = [0.0] * args.chunk_length
            coord_counts = [0] * args.chunk_length
            row_count, col_count = W_before.shape
            for row_id in range(row_count):
                if not is_watermark_row(map_args, layer_id, row_id, name):
                    continue
                for col_id in range(col_count):
                    if mask is not None and not bool(mask[row_id, col_id].item()):
                        continue
                    if float(W_before[row_id, col_id].item()) == 0.0:
                        continue
                    use_weight, bit_pos, coeff = get_projection_bit_mapping(
                        map_args, layer_id, row_id, col_id, args.chunk_length, name
                    )
                    if not use_weight:
                        continue
                    projections_before[bit_pos] += coeff * float(W_before[row_id, col_id].item())
                    projections_after[bit_pos] += coeff * float(W_after[row_id, col_id].item())
                    coord_counts[bit_pos] += 1
            for bit_pos in range(args.chunk_length):
                bit_idx = chunk_id * args.chunk_length + bit_pos
                target_bit = watermark_bits[bit_idx]
                target_sign = 1.0 if target_bit == "1" else -1.0
                z_before = projections_before[bit_pos]
                z_after = projections_after[bit_pos]
                m_before = target_sign * z_before
                m_after = target_sign * z_after
                if coord_counts[bit_pos] == 0:
                    empty_groups += 1
                    continue
                before_sum += m_before
                after_sum += m_after
                drop_sum += (m_before - m_after)
                retention_sum += m_after / (m_before + eps)
                positive_before += int(m_before > 0)
                positive_after += int(m_after > 0)
                sign_flip += int((z_before > 0) != (z_after > 0))
                count += 1
                records.append({
                    "block_idx": block_idx,
                    "layer_id": layer_id,
                    "layer_name": name,
                    "selected_matrix_id": selected_matrix_id,
                    "chunk_id": chunk_id,
                    "bit_pos": bit_pos,
                    "bit_idx": bit_idx,
                    "target_bit": target_bit,
                    "coord_count": coord_counts[bit_pos],
                    "z_before": z_before,
                    "z_after": z_after,
                    "margin_before": m_before,
                    "margin_after": m_after,
                    "margin_drop": m_before - m_after,
                    "margin_retention": m_after / (m_before + eps),
                    "sign_flip": bool((z_before > 0) != (z_after > 0)),
                })
            layer_id += 1
            selected_matrix_id += 1
    return {
        "scope": scope,
        "bit_records_count": count,
        "empty_groups": empty_groups,
        "margin_before_mean": safe_mean(before_sum, count),
        "margin_after_mean": safe_mean(after_sum, count),
        "margin_drop_mean": safe_mean(drop_sum, count),
        "margin_retention_mean": safe_mean(retention_sum, count),
        "positive_margin_rate_before": safe_mean(positive_before, count),
        "positive_margin_rate_after": safe_mean(positive_after, count),
        "sign_flip_rate": safe_mean(sign_flip, count),
        "per_bit": records,
    }

def load_utility_json(paths):
    merged = []
    for path in paths or []:
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, dict) and "results" in data:
            for row in data["results"]:
                item = dict(row)
                item["source_json"] = path
                merged.append(item)
        else:
            merged.append({"source_json": path, "raw": data})
    return merged

def _path_tokens(path):
    norm_path = os.path.normpath(path)
    return {norm_path, os.path.basename(norm_path)}

def _utility_matches(row, model_path):
    tokens = _path_tokens(model_path)
    row_path = row.get("model_path")
    row_name = row.get("model_name")
    if row_path and os.path.normpath(row_path) in tokens:
        return True
    if row_name and row_name in tokens:
        return True
    return False

def attach_utility_results(results, specs, utility_rows):
    if not utility_rows:
        return
    for result, spec in zip(results, specs):
        result["utility"] = {
            "watermarked_model": [
                row for row in utility_rows
                if isinstance(row, dict) and _utility_matches(row, spec.watermarked_model)
            ],
            "finetuned_model": [
                row for row in utility_rows
                if isinstance(row, dict) and _utility_matches(row, spec.finetuned_model)
            ],
        }

def summarize_method(spec, args):
    print("=" * 80)
    print(f"Evaluating method: {spec.name}")
    print(f"  subspace: {spec.subspace_path}")
    print(f"  watermarked: {spec.watermarked_model}")
    print(f"  finetuned: {spec.finetuned_model}")
    masks, mapping_config = load_masks(spec.subspace_path)
    wm_model = load_model(spec.watermarked_model, args.device)
    ft_model = load_model(spec.finetuned_model, args.device)
    try:
        drift_all = compute_drift_for_scope(
            wm_model, ft_model, masks, args, selected_layers_only=False
        )
        drift_watermark_layers = compute_drift_for_scope(
            wm_model, ft_model, masks, args, selected_layers_only=True
        )
        margin = {}
        if args.margin_scope in ("detector", "both"):
            margin["detector"] = compute_projection_margin_for_scope(
                wm_model, ft_model, masks, args, scope="detector"
            )
        if args.margin_scope in ("mask", "both"):
            margin["mask"] = compute_projection_margin_for_scope(
                wm_model, ft_model, masks, args, scope="mask"
            )
    finally:
        del wm_model
        del ft_model
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    return {
        "method": spec.name,
        "paths": {
            "subspace_path": spec.subspace_path,
            "watermarked_model": spec.watermarked_model,
            "finetuned_model": spec.finetuned_model,
        },
        "mapping_config": mapping_config,
        "drift_all_masked_matrices": drift_all,
        "drift_watermark_selected_matrices": drift_watermark_layers,
        "projection_margin": margin,
    }

def compact_float(value):
    if value is None:
        return "NA"
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return str(value)
    return f"{value:.6g}"

def print_summary(results):
    print("\n" + "=" * 80)
    print("Stable-space metric summary")
    header = [
        "method",
        "sel_drift(wm_layers)",
        "drift_ratio(wm_layers)",
        "det_margin_ret",
        "det_margin_drop",
        "det_sign_flip",
        "mask_margin_ret",
        "mask_margin_drop",
        "mask_sign_flip",
    ]
    print("\t".join(header))
    for item in results:
        drift = item["drift_watermark_selected_matrices"]
        det = item["projection_margin"].get("detector", {})
        mask = item["projection_margin"].get("mask", {})
        row = [
            item["method"],
            compact_float(drift.get("selected_drift")),
            compact_float(drift.get("drift_ratio")),
            compact_float(det.get("margin_retention_mean")),
            compact_float(det.get("margin_drop_mean")),
            compact_float(det.get("sign_flip_rate")),
            compact_float(mask.get("margin_retention_mean")),
            compact_float(mask.get("margin_drop_mean")),
            compact_float(mask.get("sign_flip_rate")),
        ]
        print("\t".join(row))

def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate stable-space ablation metrics for LineageMark."
    )
    parser.add_argument(
        "--method_spec",
        action="append",
        type=parse_method_spec,
        default=None,
        help=(
            "Repeatable: name:subspace_path:watermarked_model:finetuned_model. "
            "Defaults to the prepared full/fisher_only/ca_only OPT-125M ablation paths."
        ),
    )
    parser.add_argument("--hidden_size", type=int, default=DEFAULT_HIDDEN_SIZE)
    parser.add_argument("--password", type=str, default=DEFAULT_PASSWORD)
    parser.add_argument("--watermark", type=str, default=DEFAULT_WATERMARK)
    parser.add_argument("--gamma_1", type=int, default=DEFAULT_GAMMA_1)
    parser.add_argument("--xi", type=int, default=DEFAULT_XI, choices=[2, 4])
    parser.add_argument("--chunk_length", type=int, default=DEFAULT_CHUNK_LENGTH, choices=[8])
    parser.add_argument(
        "--margin_scope",
        choices=["detector", "mask", "both"],
        default="both",
        help="detector uses key-reconstructed coordinates; mask restricts to W_mask coordinates.",
    )
    parser.add_argument("--margin_eps", type=float, default=1e-12)
    parser.add_argument(
        "--device",
        default="cpu",
        help="Model loading device. Use cpu for parameter-only metrics or auto/cuda for large local RAM pressure.",
    )
    parser.add_argument(
        "--utility_json",
        action="append",
        default=None,
        help=(
            "Optional ppl_result.py JSON files to merge into the output for model utility comparison. "
            "Defaults to outputs/ppl_diff_space_val.json."
        ),
    )
    parser.add_argument(
        "--output",
        default=DEFAULT_OUTPUT,
        help="Path to save JSON results.",
    )
    args = parser.parse_args()
    if args.method_spec is None:
        args.method_spec = list(DEFAULT_METHOD_SPECS)
    if args.utility_json is None:
        args.utility_json = list(DEFAULT_UTILITY_JSONS)
    return args

def main():
    args = parse_args()
    results = [summarize_method(spec, args) for spec in args.method_spec]
    utility_results = load_utility_json(args.utility_json)
    attach_utility_results(results, args.method_spec, utility_results)
    output = {
        "config": {
            "hidden_size": args.hidden_size,
            "watermark": args.watermark,
            "gamma_1": args.gamma_1,
            "xi": args.xi,
            "chunk_length": args.chunk_length,
            "margin_scope": args.margin_scope,
            "device": args.device,
        },
        "methods": results,
        "utility_results": utility_results,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)
    print_summary(results)
    if utility_results:
        print(f"\nMerged {len(utility_results)} utility rows from {len(args.utility_json)} JSON file(s).")
    print(f"\nSaved metrics to: {args.output}")

if __name__ == "__main__":
    main()
