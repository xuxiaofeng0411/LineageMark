import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../")

import numpy as np
import torch

from lib_multi.ellmark_extract_multi import extract_watermark
from lib_multi.utils_multi import format_time, get_llm


def print_config(args):
    print("Config parameters:")
    rows = [
        ("+ model: ", args.model),
        ("+ watermark: ", args.watermark),
        ("+ chunk_length: ", args.chunk_length),
        ("+ password: ", args.password),
        ("+ random seed: ", args.seed),
        ("+ gamma_layer: ", args.gamma_1),
        ("+ xi: ", args.xi),
        ("+ position_num: ", args.position_num),
        ("+ extract mode: ", args.mode),
        ("+ threshold_layer_acc: ", args.threshold_layer_acc),
        ("+ threshold_acc: ", args.threshold_acc),
    ]
    for label, value in rows:
        print(f"{label}{value}")


def build_parser():
    parser = argparse.ArgumentParser(description="ELLMark baseline watermark extraction")
    parser.add_argument("--model", type=str, required=True, help="Watermarked model path")
    parser.add_argument("--hidden_size", type=int, required=True, help="Hidden size of the model")
    parser.add_argument("--password", type=str, required=True, help="Key to decrypt watermark positions")
    parser.add_argument("--watermark", type=str, required=True, help="Expected watermark content")
    parser.add_argument("--chunk_length", type=int, default=8, choices=[8], help="Watermark chunk length")
    parser.add_argument("--gamma_1", type=int, default=2, help="Layer selection modulus")
    parser.add_argument("--xi", type=int, default=2, choices=[2, 4], help="Last xi bits scanned")
    parser.add_argument("--position_num", type=int, default=12, choices=[6, 12], help="Leading weight bits for position hash")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--mode", type=str, default="simple", choices=["simple", "robust"], help="Extraction mode")
    parser.add_argument("--threshold_layer_acc", type=float, default=0.70, help="Qualified layer threshold")
    parser.add_argument("--threshold_acc", type=float, default=0.99, help="Early-stop total threshold")
    return parser


def main():
    args = build_parser().parse_args()
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    print_config(args)

    print("*" * 30)
    print(f"Loading llm model: {args.model}...")
    model = get_llm(args.model)
    model.eval()

    print("ELLMark extract starts.")
    start_time = time.time()
    extracted_watermark_acc = extract_watermark(args, model)
    print("*" * 30)
    print("ELLMark Extract Done!")
    print(f"Extract ACC: {extracted_watermark_acc}")
    format_time(time.time() - start_time, "Total extract")


if __name__ == "__main__":
    main()
