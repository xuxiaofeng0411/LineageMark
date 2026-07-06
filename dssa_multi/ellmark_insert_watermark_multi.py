import argparse
import os
import sys
import time

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../")

import numpy as np
import torch
from transformers import AutoTokenizer

from lib_multi.ellmark_insert_multi import insert_watermark
from lib_multi.utils_multi import format_time, get_llm


def _add_dataset_arg(parser):
    parser.add_argument(
        "--calibdation_dataset",
        "--calibration_dataset",
        dest="calibdation_dataset",
        type=str,
        default="wikitext2",
        choices=["c4", "wikitext2"],
        help="Calibration dataset name; LineageMark data_multi currently maps this to wikitext2.",
    )


def print_config(args):
    print("Config parameters:")
    rows = [
        ("+ model: ", args.model),
        ("+ hidden_size: ", args.hidden_size),
        ("+ watermark: ", args.watermark),
        ("+ password: ", args.password),
        ("+ calibdation data: ", args.calibdation_dataset),
        ("+ num of samples: ", args.nsamples),
        ("+ random seed: ", args.seed),
        ("+ select_ratio: ", args.select_ratio),
        ("+ gamma_layer: ", args.gamma_1),
        ("+ xi: ", args.xi),
        ("+ position_num: ", args.position_num),
        ("+ delta: ", args.delta),
        ("+ save_model_path: ", args.save_model),
    ]
    for label, value in rows:
        print(f"{label}{value}")


def build_parser():
    parser = argparse.ArgumentParser(description="ELLMark baseline watermark insertion")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--hidden_size", type=int, required=True, help="Hidden size of the model")
    parser.add_argument("--password", type=str, required=True, help="Key to encrypt watermark positions")
    parser.add_argument("--watermark", type=str, required=True, help="Watermark content string")
    parser.add_argument("--gamma_1", type=int, default=2, help="Layer selection modulus")
    parser.add_argument("--xi", type=int, default=2, choices=[2, 4], help="Last xi bits may be changed")
    parser.add_argument("--position_num", type=int, default=12, choices=[6, 12], help="Leading weight bits for position hash")
    parser.add_argument("--delta", type=int, default=20, help="Minimum redundancy of bits to reverse")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration samples")
    parser.add_argument("--select_ratio", type=float, default=0.75, help="Fraction of weights selected as candidates")
    parser.add_argument("--save_model", type=str, required=True, help="Path to save the watermarked model")
    _add_dataset_arg(parser)
    return parser


def main():
    args = build_parser().parse_args()
    np.random.seed(args.seed)
    torch.random.manual_seed(args.seed)
    print_config(args)

    print("*" * 30)
    print(f"Loading llm model and tokenizer: {args.model}...")
    model = get_llm(args.model)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model, use_fast=False)

    device = torch.device("cuda")
    print("ELLMark insert starts.")
    start_time = time.time()
    change_weight_num, total_weight_num = insert_watermark(
        args,
        model,
        tokenizer,
        device,
        dataset_name=args.calibdation_dataset,
    )

    model.save_pretrained(args.save_model, safe_serialization=False)
    tokenizer.save_pretrained(args.save_model)
    print(f"model saved in {args.save_model}.")

    print("*" * 30)
    print("ELLMark Insert Done!")
    format_time(time.time() - start_time, "Total insert")
    print(f"Total change weights num: {change_weight_num}\tratio:{change_weight_num / total_weight_num}")


if __name__ == "__main__":
    main()
