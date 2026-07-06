import argparse
import os
import sys
import time

os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../')

import numpy as np
import torch
from transformers import AutoTokenizer

from lib_multi.lineagemark_insert_multi import insert_watermark
from lib_multi.subspace_multi import select_subspace
from lib_multi.utils_multi import format_time, get_llm


def _cpu_recursive(obj):
    if isinstance(obj, torch.Tensor):
        return obj.cpu()
    if isinstance(obj, dict):
        return {key: _cpu_recursive(value) for key, value in obj.items()}
    if isinstance(obj, list):
        return [_cpu_recursive(value) for value in obj]
    return obj


def _add_arguments(parser, specs):
    for flags, kwargs in specs:
        parser.add_argument(*flags, **kwargs)


def _build_parser():
    parser = argparse.ArgumentParser(
        description="DSSA: Matrix-level Subspace Selection + Watermark Insertion"
    )

    dssa_arguments = [
        (('--k',), dict(type=int, default=64,
                       help='Number of subspace directions per matrix (default: 64)')),
        (('--tau_lower',), dict(type=float, default=0.1,
                                help='Spectral truncation lower bound (default: 0.1)')),
        (('--tau_upper',), dict(type=float, default=0.9,
                                help='Spectral truncation upper bound (default: 0.9)')),
        (('--epsilon',), dict(type=float, default=1e-6,
                              help='GEVP regularization (default: 1e-6)')),
        (('--save_subspace',), dict(type=str, default=None,
                                    help='Optional path to save matrix subspace .pt file')),
        (('--dssa_block_chunk',), dict(type=int, default=0,
                                       help='Blocks to process per calibration pass; 0 means all blocks')),
        (('--dssa_calib_batch_size',), dict(type=int, default=1,
                                            help='Calibration samples per forward/backward pass; 1 preserves old memory use')),
        (('--subspace_method',), dict(type=str, default='full',
                                      choices=['full', 'fisher_only', 'ca_only'],
                                      help='Stable-space solver: full uses Fisher+C_A GEVP; fisher_only uses top Fisher EVP; ca_only uses bottom C_A EVP')),
    ]
    watermark_arguments = [
        (('--hidden_size',), dict(type=int, required=True,
                                  help='Hidden size of the model')),
        (('--password',), dict(type=str, required=True,
                               help='Key to encrypt watermark positions')),
        (('--watermark',), dict(type=str, required=True,
                                help='Watermark content string')),
        (('--gamma_1',), dict(type=int, default=2,
                              help='Layer selection modulus (default: 2)')),
        (('--xi',), dict(type=int, default=2, choices=[2, 4],
                         help='Last xi bits modifiable. fp16=4, int8=2.')),
        (('--position_num',), dict(type=int, default=12, choices=[6, 12],
                                   help='Leading weight bits used only by bitflip ELLMark-compatible mapping.')),
        (('--delta',), dict(type=int, default=20,
                            help='Minimum redundancy of bits to reverse (default: 20)')),
        (('--wm_method',), dict(type=str, default='projection',
                                choices=['projection', 'mean_diff', 'bitflip'],
                                help='Watermark embedding method (default: projection)')),
        (('--mean_margin',), dict(type=float, default=0.02,
                                  help='Target mean difference margin for mean_diff watermark')),
        (('--projection_margin',), dict(type=float, default=0.5,
                                        help='Target signed projection margin for projection watermark')),
        (('--projection_max_update',), dict(type=float, default=0.0,
                                            help='Optional max update per selected weight; 0 disables clipping')),
        (('--data_independent_extract',), dict(action='store_true',
                                               help='Embed a key-recoverable detector while editing only DSSA-selected weights')),
    ]
    common_arguments = [
        (('--model',), dict(type=str, required=True,
                            help='Path to pretrained model')),
        (('--nsamples',), dict(type=int, default=512,
                               help='Number of calibration samples (default: 512)')),
        (('--seqlen',), dict(type=int, default=2048,
                             help='Sequence length for calibration (default: 2048)')),
        (('--seed',), dict(type=int, default=42,
                           help='Random seed (default: 42)')),
        (('--select_ratio',), dict(type=float, default=0.75,
                                   help='Fraction of matrix coordinates to select globally in W_mask (default: 0.75)')),
        (('--dataset',), dict(type=str, default='wikitext2',
                              choices=['wikitext2'],
                              help='Calibration dataset (default: wikitext2)')),
        (('--save_model',), dict(type=str, required=True,
                                 help='Path to save the watermarked model')),
    ]

    for group in (dssa_arguments, watermark_arguments, common_arguments):
        _add_arguments(parser, group)
    return parser


def print_config(args):
    print("Config parameters:")
    config_rows = [
        ("  + model:              ", args.model),
        ("  + k (subspace dim):   ", args.k),
        ("  + nsamples:           ", args.nsamples),
        ("  + seqlen:             ", args.seqlen),
        ("  + seed:               ", args.seed),
        ("  + tau_lower:          ", args.tau_lower),
        ("  + tau_upper:          ", args.tau_upper),
        ("  + epsilon:            ", args.epsilon),
        ("  + select_ratio:       ", args.select_ratio),
        ("  + dssa_block_chunk:   ", args.dssa_block_chunk),
        ("  + dssa_calib_batch_size: ", args.dssa_calib_batch_size),
        ("  + subspace_method:    ", args.subspace_method),
        ("  + dataset:            ", args.dataset),
        ("  + hidden_size:        ", args.hidden_size),
        ("  + watermark:          ", args.watermark),
        ("  + password:           ", args.password),
        ("  + gamma_1:            ", args.gamma_1),
        ("  + xi:                 ", args.xi),
        ("  + position_num:       ", args.position_num),
        ("  + delta:              ", args.delta),
        ("  + wm_method:          ", args.wm_method),
        ("  + mean_margin:        ", args.mean_margin),
        ("  + projection_margin:  ", args.projection_margin),
        ("  + projection_max_update: ", args.projection_max_update),
        ("  + data_independent_extract: ", args.data_independent_extract),
        ("  + save_model:         ", args.save_model),
    ]
    for label, value in config_rows:
        print(f"{label}{value}")
    if args.save_subspace:
        print(f"  + save_subspace:      {args.save_subspace}")


def _seed_everything(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def _load_model_and_tokenizer(model_name):
    print("*" * 60)
    print(f"Loading llm model and tokenizer: {model_name}...")
    model = get_llm(model_name)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=True)
    return model, tokenizer


def _run_subspace_selection(args, model, tokenizer, device):
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
        is_8bit="8bit" in args.model,
        device=device,
        dataset_name=args.dataset,
        seed=args.seed,
        dssa_block_chunk=args.dssa_block_chunk,
        calib_batch_size=args.dssa_calib_batch_size,
        subspace_method=args.subspace_method,
    )
    format_time(time.time() - start_time, "Matrix-level DSSA subspace selection")
    return result


def _watermark_config(args):
    return {
        'k': args.k,
        'tau_lower': args.tau_lower,
        'tau_upper': args.tau_upper,
        'epsilon': args.epsilon,
        'select_ratio': args.select_ratio,
        'dssa_block_chunk': args.dssa_block_chunk,
        'dssa_calib_batch_size': args.dssa_calib_batch_size,
        'subspace_method': args.subspace_method,
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


def _save_subspace_if_requested(args, result, all_masks):
    if not args.save_subspace:
        return

    save_dict = _cpu_recursive(result)
    save_dict['all_layer_masks'] = _cpu_recursive(all_masks)
    save_dict['mapping_version'] = (
        'dssa-matrix-key-detector-v1' if args.data_independent_extract else 'dssa-matrix-v1'
    )
    save_dict['watermark_config'] = _watermark_config(args)
    torch.save(save_dict, args.save_subspace)
    print(f"Matrix subspaces and watermark map saved to: {args.save_subspace}")


def _insert_with_dssa(args, model, tokenizer, device, all_masks):
    print("\n" + "=" * 60)
    print("PHASE 2: Watermark Insertion using matrix-level DSSA W_mask")
    print("=" * 60)
    start_time = time.time()
    change_weight_num, total_weight_num = insert_watermark(
        args, model, tokenizer, device,
        dataset_name=args.dataset,
        dssa_layer_masks=all_masks,
    )
    format_time(time.time() - start_time, "Watermark insertion")
    return change_weight_num, total_weight_num


def _save_watermarked_model(args, model, tokenizer):
    print("*" * 60)
    model.save_pretrained(args.save_model, safe_serialization=False)
    tokenizer.save_pretrained(args.save_model)
    print(f"Watermarked model saved to: {args.save_model}")


def main():
    args = _build_parser().parse_args()
    _seed_everything(args.seed)
    print_config(args)

    model, tokenizer = _load_model_and_tokenizer(args.model)
    device = torch.device("cuda")

    subspace_result = _run_subspace_selection(args, model, tokenizer, device)
    all_masks = subspace_result['all_layer_masks']
    _save_subspace_if_requested(args, subspace_result, all_masks)

    change_weight_num, total_weight_num = _insert_with_dssa(
        args, model, tokenizer, device, all_masks
    )
    _save_watermarked_model(args, model, tokenizer)

    print("*" * 60)
    print("Matrix-level DSSA watermark insertion complete!")
    print(f"Total changed weights: {change_weight_num} / {total_weight_num} "
          f"({change_weight_num / total_weight_num * 100:.2f}%)")


if __name__ == '__main__':
    main()
