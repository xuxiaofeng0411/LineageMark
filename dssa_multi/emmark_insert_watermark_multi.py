import argparse
import os
import sys
import time

os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)) + "/../")

import numpy as np
import torch
from transformers import AutoTokenizer

from lib_multi.emmark_insert_multi import insert_watermark
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
        ("+ hidden size: ", args.hidden_size),
        ("+ watermark: ", args.watermark),
        ("+ calibdation data: ", args.calibdation_dataset),
        ("+ num of samples: ", args.nsamples),
        ("+ random seed: ", args.seed),
        ("+ candidate rate: ", args.candidate_rate),
        ("+ modify rate: ", args.modify_rate),
        ("+ save model path: ", args.save_model),
    ]
    for label, value in rows:
        print(f"{label}{value}")


def build_parser():
    parser = argparse.ArgumentParser(description="EmMark baseline watermark insertion")
    parser.add_argument("--model", type=str, required=True, help="Model path")
    parser.add_argument("--hidden_size", type=int, required=True, help="Hidden size of the model")
    parser.add_argument("--watermark", type=str, required=True, help="Watermark content string")
    parser.add_argument("--seed", type=int, default=100, help="Seed for choosing weights to modify")
    parser.add_argument("--nsamples", type=int, default=128, help="Number of calibration samples")
    parser.add_argument("--candidate_rate", type=int, default=60, help="|B_c|/(|B|/n)")
    parser.add_argument("--modify_rate", type=float, default=0.75, help="Rate of hidden size to modify")
    parser.add_argument("--save_model", type=str, required=True, help="Path to save the watermarked model and total_indices.pt")
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
    tokenizer = AutoTokenizer.from_pretrained(args.model)

    device = torch.device("cuda")
    print("EmMark insert starts.")
    start_time = time.time()
    change_weight_num, total_weight_num, total_indices = insert_watermark(
        args,
        model,
        tokenizer,
        device,
        dataset_name=args.calibdation_dataset,
    )

    model.save_pretrained(args.save_model, safe_serialization=False)
    tokenizer.save_pretrained(args.save_model)
    torch.save(total_indices.to(torch.int32).cpu(), os.path.join(args.save_model, "total_indices.pt"))
    print(f"model saved in {args.save_model}.")

    print("*" * 30)
    print("EmMark Insert Done!")
    format_time(time.time() - start_time, "Total insert")
    print(f"Total change weights num: {change_weight_num}\tratio:{change_weight_num / total_weight_num}")


if __name__ == "__main__":
    main()
