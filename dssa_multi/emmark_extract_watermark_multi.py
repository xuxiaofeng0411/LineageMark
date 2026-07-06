import argparse
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../")

import numpy as np
import torch

from lib_multi.emmark_extract_multi import extract_watermark
from lib_multi.utils_multi import format_time, get_llm


def print_config(args):
    print("Config parameters:")
    rows = [
        ("+ origin model: ", args.model),
        ("+ inserted model: ", args.inserted_model),
        ("+ hidden size: ", args.hidden_size),
        ("+ watermark: ", args.watermark),
        ("+ random seed: ", args.seed),
        ("+ candidate rate: ", args.candidate_rate),
        ("+ modify rate: ", args.modify_rate),
    ]
    for label, value in rows:
        print(f"{label}{value}")


def build_parser():
    parser = argparse.ArgumentParser(description="EmMark baseline watermark extraction")
    parser.add_argument("--model", type=str, required=True, help="Original model path")
    parser.add_argument("--inserted_model", type=str, required=True, help="Watermarked model path")
    parser.add_argument("--hidden_size", type=int, required=True, help="Hidden size of the model")
    parser.add_argument("--watermark", type=str, required=True, help="Expected watermark content")
    parser.add_argument("--seed", type=int, default=100, help="Seed for choosing weights to modify")
    parser.add_argument("--candidate_rate", type=int, default=60, help="|B_c|/(|B|/n)")
    parser.add_argument("--modify_rate", type=float, default=0.75, help="Rate of hidden size to modify")
    return parser


def main():
    args = build_parser().parse_args()
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    print_config(args)

    print("*" * 30)
    print(f"Loading original model: {args.model}...")
    model = get_llm(args.model)
    model.eval()
    print(f"Loading inserted model: {args.inserted_model}...")
    inserted_model = get_llm(args.inserted_model)
    inserted_model.eval()

    indices_path = os.path.join(args.inserted_model, "total_indices.pt")
    indices = torch.load(indices_path, map_location="cpu")

    print("EmMark extract starts.")
    start_time = time.time()
    extracted_watermark_acc = extract_watermark(args, model, inserted_model, indices)
    print("*" * 30)
    print("EmMark Extract Done!")
    print(f"Extract ACC: {extracted_watermark_acc}")
    format_time(time.time() - start_time, "Total extract")


if __name__ == "__main__":
    main()
