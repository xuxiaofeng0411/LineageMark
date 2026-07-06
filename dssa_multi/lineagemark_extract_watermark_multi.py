import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + '/../')

import numpy as np
import torch

from lib_multi.lineagemark_extract_multi import extract_watermark
from lib_multi.utils_multi import format_time, get_llm


def _load_pt_cpu(path):
    try:
        return torch.load(path, map_location='cpu', weights_only=False)
    except TypeError:
        return torch.load(path, map_location='cpu')


def load_watermark_map(path):
    if not path:
        return None

    saved = _load_pt_cpu(path)
    if isinstance(saved, dict) and 'all_layer_masks' in saved:
        print(f"Loaded watermark map from: {path}")
        return saved['all_layer_masks']
    if isinstance(saved, dict) and all(isinstance(k, int) for k in saved.keys()):
        print(f"Loaded raw watermark map from: {path}")
        return saved
    raise ValueError(
        f"{path} does not contain all_layer_masks. Re-run insertion with --save_subspace."
    )


def _add_arguments(parser, specs):
    for flags, kwargs in specs:
        parser.add_argument(*flags, **kwargs)


def _build_parser():
    parser = argparse.ArgumentParser(
        description="DSSA: Extract watermark from DSSA-watermarked model"
    )
    argument_specs = [
        (('--model',), dict(type=str, required=True,
                            help='Path to DSSA-watermarked model')),
        (('--hidden_size',), dict(type=int, required=True,
                                  help='Hidden size of the model')),
        (('--password',), dict(type=str, required=True,
                               help='Key to decrypt watermark positions')),
        (('--watermark',), dict(type=str, required=True,
                                help='Expected watermark content')),
        (('--chunk_length',), dict(type=int, default=8, choices=[8],
                                   help='Watermark chunk length (default: 8)')),
        (('--gamma_1',), dict(type=int, default=2,
                              help='Layer selection modulus (default: 2)')),
        (('--xi',), dict(type=int, default=2, choices=[2, 4],
                         help='Last xi bits scanned. fp16=4, int8=2.')),
        (('--position_num',), dict(type=int, default=12, choices=[6, 12],
                                   help='Leading weight bits used only by bitflip ELLMark-compatible mapping.')),
        (('--wm_method',), dict(type=str, default='projection',
                                choices=['projection', 'mean_diff', 'bitflip'],
                                help='Watermark extraction method (default: projection)')),
        (('--mean_margin',), dict(type=float, default=0.02,
                                  help='Expected mean difference margin, used for logging compatibility')),
        (('--projection_margin',), dict(type=float, default=0.5,
                                        help='Expected projection margin, used for logging compatibility')),
        (('--data_independent_extract',), dict(action='store_true',
                                               help='Ignore watermark_map and extract from key-recoverable coordinates only')),
        (('--seed',), dict(type=int, default=42,
                           help='Random seed (default: 42)')),
        (('--mode',), dict(type=str, default='simple', choices=['simple', 'robust'],
                           help='Extract mode: simple (faster) or robust (for fine-tuned models)')),
        (('--threshold_layer_acc',), dict(type=float, default=0.70,
                                          help='Deprecated: ignored by blind extraction')),
        (('--threshold_acc',), dict(type=float, default=0.99,
                                    help='Deprecated: ignored by blind extraction')),
        (('--watermark_map',), dict(type=str, default=None,
                                    help='Optional .pt file saved by insertion --save_subspace')),
    ]
    _add_arguments(parser, argument_specs)
    return parser


def print_config(args):
    print(f'Config parameters:')
    rows = [
        ("+ model: ", args.model),
        ("+ hidden_size: ", args.hidden_size),
        ("+ watermark: ", args.watermark),
        ("+ chunk_length: ", args.chunk_length),
        ("+ password: ", args.password),
        ("+ random seed: ", args.seed),
        ("+ gamma_layer: ", args.gamma_1),
        ("+ xi: ", args.xi),
        ("+ position_num: ", args.position_num),
        ("+ wm_method: ", args.wm_method),
        ("+ mean_margin: ", args.mean_margin),
        ("+ projection_margin: ", args.projection_margin),
        ("+ data_independent_extract: ", args.data_independent_extract),
        ("+ extract mode: ", args.mode),
        ("+ threshold_layer_acc: ", f"{args.threshold_layer_acc} (ignored by blind extraction)"),
        ("+ threshold_acc: ", f"{args.threshold_acc} (ignored by blind extraction)"),
        ("+ watermark_map: ", args.watermark_map),
    ]
    for prefix, value in rows:
        print(f"{prefix}{value}")


def _seed_everything(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


def _load_marked_model(model_name):
    print("*" * 60)
    print(f"Loading DSSA-watermarked model: {model_name}...")
    model = get_llm(model_name)
    model.eval()
    return model


def _resolve_masks(args):
    if args.data_independent_extract:
        return None
    return load_watermark_map(args.watermark_map)


def _run_extraction(args, model, dssa_layer_masks):
    print("DSSA extract starts.")
    start_time = time.time()
    extracted_watermark_acc = extract_watermark(args, model, dssa_layer_masks=dssa_layer_masks)
    print("*" * 60)
    print('DSSA Extract Done!')
    print(f'Extract ACC: {extracted_watermark_acc}')
    format_time(time.time() - start_time, "Total extract")


def main():
    args = _build_parser().parse_args()
    _seed_everything(args.seed)
    print_config(args)

    model = _load_marked_model(args.model)
    dssa_layer_masks = _resolve_masks(args)
    _run_extraction(args, model, dssa_layer_masks)


if __name__ == '__main__':
    main()
