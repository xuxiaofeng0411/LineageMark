import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../')

import argparse
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
import time
import numpy as np
import torch
from transformers import AutoTokenizer

from lib_multi.subspace_multi import select_subspace
from lib_multi.insert_multi import insert_watermark
from lib_multi.utils_multi import get_llm, format_time


def _cpu_recursive(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    elif isinstance(obj, dict):
        return {k: _cpu_recursive(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_cpu_recursive(v) for v in obj]
    return obj


def print_config(args):
    print("Config parameters:")
    print(f"  + model:              {args.model}")
    print(f"  + k (subspace dim):   {args.k}")
    print(f"  + nsamples:           {args.nsamples}")
    print(f"  + seqlen:             {args.seqlen}")
    print(f"  + seed:               {args.seed}")
    print(f"  + tau_lower:          {args.tau_lower}")
    print(f"  + tau_upper:          {args.tau_upper}")
    print(f"  + epsilon:            {args.epsilon}")
    print(f"  + select_ratio:       {args.select_ratio}")
    print(f"  + dssa_block_chunk:   {args.dssa_block_chunk}")
    print(f"  + dssa_calib_batch_size: {args.dssa_calib_batch_size}")
    print(f"  + dataset:            {args.dataset}")
    print(f"  + hidden_size:        {args.hidden_size}")
    print(f"  + watermark:          {args.watermark}")
    print(f"  + password:           {args.password}")
    print(f"  + gamma_1:            {args.gamma_1}")
    print(f"  + xi:                 {args.xi}")
    print(f"  + position_num:       {args.position_num}")
    print(f"  + delta:              {args.delta}")
    print(f"  + wm_method:          {args.wm_method}")
    print(f"  + mean_margin:        {args.mean_margin}")
    print(f"  + projection_margin:  {args.projection_margin}")
    print(f"  + projection_max_update: {args.projection_max_update}")
    print(f"  + data_independent_extract: {args.data_independent_extract}")
    print(f"  + save_model:         {args.save_model}")
    if args.save_subspace:
        print(f"  + save_subspace:      {args.save_subspace}")


def main():
    parser = argparse.ArgumentParser(
        description="DSSA: Matrix-level Subspace Selection + Watermark Insertion"
    )

    # ---- DSSA subspace selection args ----
    parser.add_argument('--k', type=int, default=64,
                        help='Number of subspace directions per matrix (default: 64)')
    parser.add_argument('--tau_lower', type=float, default=0.1,
                        help='Spectral truncation lower bound (default: 0.1)')
    parser.add_argument('--tau_upper', type=float, default=0.9,
                        help='Spectral truncation upper bound (default: 0.9)')
    parser.add_argument('--epsilon', type=float, default=1e-6,
                        help='GEVP regularization (default: 1e-6)')
    parser.add_argument('--save_subspace', type=str, default=None,
                        help='Optional path to save matrix subspace .pt file')
    parser.add_argument('--dssa_block_chunk', type=int, default=0,
                        help='Blocks to process per calibration pass; 0 means all blocks')
    parser.add_argument('--dssa_calib_batch_size', type=int, default=1,
                        help='Calibration samples per forward/backward pass; 1 preserves old memory use')

    # ---- Watermark insertion args ----
    parser.add_argument('--hidden_size', type=int, required=True,
                        help='Hidden size of the model')
    parser.add_argument('--password', type=str, required=True,
                        help='Key to encrypt watermark positions')
    parser.add_argument('--watermark', type=str, required=True,
                        help='Watermark content string')
    parser.add_argument('--gamma_1', type=int, default=2,
                        help='Layer selection modulus (default: 2)')
    parser.add_argument('--xi', type=int, default=2, choices=[2, 4],
                        help='Last xi bits modifiable. fp16=4, int8=2.')
    parser.add_argument('--position_num', type=int, default=6, choices=[6, 12],
                        help='Leading bits for position hash. fp16=12, int8=6.')
    parser.add_argument('--delta', type=int, default=20,
                        help='Minimum redundancy of bits to reverse (default: 20)')
    parser.add_argument('--wm_method', type=str, default='projection',
                        choices=['projection', 'mean_diff', 'bitflip'],
                        help='Watermark embedding method (default: projection)')
    parser.add_argument('--mean_margin', type=float, default=0.02,
                        help='Target mean difference margin for mean_diff watermark')
    parser.add_argument('--projection_margin', type=float, default=0.5,
                        help='Target signed projection margin for projection watermark')
    parser.add_argument('--projection_max_update', type=float, default=0.0,
                        help='Optional max update per selected weight; 0 disables clipping')
    parser.add_argument('--data_independent_extract', action='store_true',
                        help='Embed a key-recoverable detector while editing only DSSA-selected weights')

    # ---- Common args ----
    parser.add_argument('--model', type=str, required=True,
                        help='Path to pretrained model')
    parser.add_argument('--nsamples', type=int, default=512,
                        help='Number of calibration samples (default: 512)')
    parser.add_argument('--seqlen', type=int, default=2048,
                        help='Sequence length for calibration (default: 2048)')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--select_ratio', type=float, default=0.75,
                        help='Fraction of columns to select per row in W_mask (default: 0.75)')
    parser.add_argument('--dataset', type=str, default='wikitext2',
                        choices=['wikitext2'],
                        help='Calibration dataset (default: wikitext2)')
    parser.add_argument('--save_model', type=str, required=True,
                        help='Path to save the watermarked model')

    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    print_config(args)

    # ------- Load model -------
    print("*" * 60)
    print(f"Loading llm model and tokenizer: {args.model}...")
    model = get_llm(args.model)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    device = torch.device("cuda")
    is_8bit = "8bit" in args.model

    # ------- Step 1: Matrix-level DSSA subspace selection -------
    print("\n" + "=" * 60)
    print("PHASE 1: Matrix-level DSSA Subspace Selection")
    print("=" * 60)
    start_time = time.time()

    result = select_subspace(
        model=model,
        tokenizer=tokenizer,
        nsamples=args.nsamples,
        seqlen=args.seqlen,
        k=args.k,
        tau_lower=args.tau_lower,
        tau_upper=args.tau_upper,
        epsilon=args.epsilon,
        select_ratio=args.select_ratio,
        is_8bit=is_8bit,
        device=device,
        dataset_name=args.dataset,
        seed=args.seed,
        dssa_block_chunk=args.dssa_block_chunk,
        calib_batch_size=args.dssa_calib_batch_size,
    )

    end_time = time.time()
    format_time(end_time - start_time, "Matrix-level DSSA subspace selection")

    all_masks = result['all_layer_masks']

    # Optionally save matrix subspaces and exact DSSA masks used later by extraction.
    if args.save_subspace:
        save_dict = _cpu_recursive(result)
        save_dict['all_layer_masks'] = _cpu_recursive(all_masks)
        save_dict['mapping_version'] = (
            'dssa-matrix-key-detector-v1' if args.data_independent_extract else 'dssa-matrix-v1'
        )
        save_dict['watermark_config'] = {
            'k': args.k,
            'tau_lower': args.tau_lower,
            'tau_upper': args.tau_upper,
            'epsilon': args.epsilon,
            'select_ratio': args.select_ratio,
            'dssa_block_chunk': args.dssa_block_chunk,
            'dssa_calib_batch_size': args.dssa_calib_batch_size,
            'hidden_size': args.hidden_size,
            'gamma_1': args.gamma_1,
            'xi': args.xi,
            'position_num': args.position_num,
            'delta': args.delta,
            'wm_method': args.wm_method,
            'mean_margin': args.mean_margin,
            'projection_margin': args.projection_margin,
            'projection_max_update': args.projection_max_update,
            'data_independent_extract': args.data_independent_extract,
            'dataset': args.dataset,
            'seed': args.seed,
            'subspace_scope': 'matrix_output',
        }
        torch.save(save_dict, args.save_subspace)
        print(f"Matrix subspaces and watermark map saved to: {args.save_subspace}")

    # ------- Step 2: Insert watermark using matrix-level DSSA masks -------
    print("\n" + "=" * 60)
    print("PHASE 2: Watermark Insertion using matrix-level DSSA W_mask")
    print("=" * 60)
    start_time = time.time()

    change_weight_num, total_weight_num = insert_watermark(
        args, model, tokenizer, device,
        dataset_name=args.dataset,
        dssa_layer_masks=all_masks,
    )

    end_time = time.time()
    format_time(end_time - start_time, "Watermark insertion")

    # ------- Save watermarked model -------
    print("*" * 60)
    model.save_pretrained(args.save_model, safe_serialization=False)
    tokenizer.save_pretrained(args.save_model)
    print(f"Watermarked model saved to: {args.save_model}")

    print("*" * 60)
    print("Matrix-level DSSA watermark insertion complete!")
    print(f"Total changed weights: {change_weight_num} / {total_weight_num} "
          f"({change_weight_num / total_weight_num * 100:.2f}%)")


if __name__ == '__main__':
    main()
