import sys, os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../')

import argparse
import os
import time
import numpy as np
import torch

from lib_multi.extract_multi import extract_watermark
from lib_multi.utils_multi import get_llm, format_time


def load_watermark_map(path):
    if not path:
        return None

    try:
        saved = torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        saved = torch.load(path, map_location='cpu')

    if isinstance(saved, dict) and 'all_layer_masks' in saved:
        print(f"Loaded watermark map from: {path}")
        return saved['all_layer_masks']

    if isinstance(saved, dict) and all(isinstance(k, int) for k in saved.keys()):
        print(f"Loaded raw watermark map from: {path}")
        return saved

    raise ValueError(
        f"{path} does not contain all_layer_masks. Re-run insertion with --save_subspace."
    )


def print_config(args):
    print(f'Config parameters:')
    print(f"+ model: {args.model}")
    print(f"+ hidden_size: {args.hidden_size}")
    print(f"+ watermark: {args.watermark}")
    print(f"+ chunk_length: {args.chunk_length}")
    print(f"+ password: {args.password}")
    print(f"+ random seed: {args.seed}")
    print(f"+ gamma_layer: {args.gamma_1}")
    print(f"+ xi: {args.xi}")
    print(f"+ position_num: {args.position_num}")
    print(f"+ wm_method: {args.wm_method}")
    print(f"+ mean_margin: {args.mean_margin}")
    print(f"+ projection_margin: {args.projection_margin}")
    print(f"+ data_independent_extract: {args.data_independent_extract}")
    print(f"+ extract mode: {args.mode}")
    print(f"+ threshold_layer_acc: {args.threshold_layer_acc} (ignored by blind extraction)")
    print(f"+ threshold_acc: {args.threshold_acc} (ignored by blind extraction)")
    print(f"+ watermark_map: {args.watermark_map}")


def main():
    parser = argparse.ArgumentParser(
        description="DSSA: Extract watermark from DSSA-watermarked model"
    )
    parser.add_argument('--model', type=str, required=True,
                        help='Path to DSSA-watermarked model')
    parser.add_argument('--hidden_size', type=int, required=True,
                        help='Hidden size of the model')
    parser.add_argument('--password', type=str, required=True,
                        help='Key to decrypt watermark positions')
    parser.add_argument('--watermark', type=str, required=True,
                        help='Expected watermark content')
    parser.add_argument('--chunk_length', type=int, default=8, choices=[8],
                        help='Watermark chunk length (default: 8)')
    parser.add_argument('--gamma_1', type=int, default=2,
                        help='Layer selection modulus (default: 2)')
    parser.add_argument('--xi', type=int, default=2, choices=[2, 4],
                        help='Last xi bits scanned. fp16=4, int8=2.')
    parser.add_argument('--position_num', type=int, default=6, choices=[6, 12],
                        help='Leading bits for position hash. fp16=12, int8=6.')
    parser.add_argument('--wm_method', type=str, default='projection',
                        choices=['projection', 'mean_diff', 'bitflip'],
                        help='Watermark extraction method (default: projection)')
    parser.add_argument('--mean_margin', type=float, default=0.02,
                        help='Expected mean difference margin, used for logging compatibility')
    parser.add_argument('--projection_margin', type=float, default=0.5,
                        help='Expected projection margin, used for logging compatibility')
    parser.add_argument('--data_independent_extract', action='store_true',
                        help='Ignore watermark_map and extract from key-recoverable coordinates only')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed (default: 42)')
    parser.add_argument('--mode', type=str, default='simple', choices=['simple', 'robust'],
                        help='Extract mode: simple (faster) or robust (for fine-tuned models)')
    parser.add_argument('--threshold_layer_acc', type=float, default=0.70,
                        help='Deprecated: ignored by blind extraction')
    parser.add_argument('--threshold_acc', type=float, default=0.99,
                        help='Deprecated: ignored by blind extraction')
    parser.add_argument('--watermark_map', type=str, default=None,
                        help='Optional .pt file saved by insertion --save_subspace')
    args = parser.parse_args()

    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)

    print_config(args)

    print("*" * 60)
    print(f"Loading DSSA-watermarked model: {args.model}...")
    model = get_llm(args.model)
    model.eval()
    dssa_layer_masks = None if args.data_independent_extract else load_watermark_map(args.watermark_map)

    print("DSSA extract starts.")
    start_time = time.time()

    extracted_watermark_acc = extract_watermark(args, model, dssa_layer_masks=dssa_layer_masks)

    print("*" * 60)
    print('DSSA Extract Done!')
    print(f'Extract ACC: {extracted_watermark_acc}')
    end_time = time.time()
    elapsed_time = end_time - start_time
    format_time(elapsed_time, "Total extract")


if __name__ == '__main__':
    main()
