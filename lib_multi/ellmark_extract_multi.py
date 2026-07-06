import hashlib
import time

import numpy as np
import torch
import tqdm

from .utils_multi import (
    find_layers,
    format_time,
    get_bit_from_weight,
    get_blocks,
    get_ellmark_bit_mapping,
    get_extract_acc,
    get_extract_chunk_acc,
    is_ellmark_watermark_row,
    qweight2weight,
    string_to_binary,
)


def _read_weight(args, module):
    if "8bit" in args.model or hasattr(module, "qweight"):
        return qweight2weight(module).data.t()
    return module.weight.data


def _bits_from_votes(ones, zeros):
    return "".join("1" if ones[idx] > zeros[idx] else "0" for idx in range(len(ones)))


def extract_watermark(args, model):
    is_robust = args.mode == "robust"
    print(f"mode: is_robust={is_robust}")

    blocks = get_blocks(model)
    threshold_acc = args.threshold_acc
    threshold_layer_acc = args.threshold_layer_acc
    real_watermark_bits = string_to_binary(args.watermark)
    bit_count = len(real_watermark_bits)
    chunk_length = args.chunk_length
    watermarks_from_layers = {"ones": [0] * bit_count, "zeors": [0] * bit_count}

    layer_id = 0
    for block_index in tqdm.tqdm(range(len(blocks)), desc="Running ELLMark extract..."):
        linear_layers = find_layers(blocks[block_index])
        for name, module in linear_layers.items():
            ones = [0] * chunk_length
            zeros = [0] * chunk_length
            one_layer_start = time.time()
            print(f"[{layer_id}]: {name}: {module}")

            layer_key = int(hashlib.md5((args.password + str(layer_id)).encode()).hexdigest(), 16)
            if not is_robust and layer_key % args.gamma_1 != 0:
                print(f"[{layer_id}]: This layer is skipped.")
                layer_id += 1
                continue

            print("==> Extract watermark from weights...")
            original_weight = _read_weight(args, module).cpu()
            ones, zeros = extract_from_weight(args, layer_id, original_weight, ones, zeros, is_robust)
            print(f"ones: {ones}")
            print(f"zeros: {zeros}")

            extracted_chunk_bits = _bits_from_votes(ones, zeros)
            print(f"=========[{layer_id}]========[{real_watermark_bits}]")
            layer_acc, chunk_id = get_extract_chunk_acc(real_watermark_bits, extracted_chunk_bits)
            print(extracted_chunk_bits)
            print(f"[{layer_id}] acc: {layer_acc}")

            format_time(time.time() - one_layer_start, "Extract of one layer")
            layer_id += 1

            if chunk_id != -1 and layer_acc > threshold_layer_acc:
                for bit_offset in range(chunk_length):
                    target_index = chunk_id * chunk_length + bit_offset
                    watermarks_from_layers["ones"][target_index] += ones[bit_offset]
                    watermarks_from_layers["zeors"][target_index] += zeros[bit_offset]
                print(f"Current ones: {watermarks_from_layers['ones']}")
                print(f"Current zeros: {watermarks_from_layers['zeors']}")

                extracted_bits = _bits_from_votes(watermarks_from_layers["ones"], watermarks_from_layers["zeors"])
                cur_total_acc = get_extract_acc(real_watermark_bits, extracted_bits)
                print(f"Current acc: {cur_total_acc}")
                if cur_total_acc >= threshold_acc:
                    print(f"Real watermark: {real_watermark_bits}")
                    print(f"Extract watermark: {extracted_bits}")
                    torch.cuda.empty_cache()
                    return cur_total_acc

    torch.cuda.empty_cache()
    extracted_bits = _bits_from_votes(watermarks_from_layers["ones"], watermarks_from_layers["zeors"])
    print(f"Real watermark: {real_watermark_bits}")
    print(f"Extract watermark: {extracted_bits}")
    return get_extract_acc(real_watermark_bits, extracted_bits)


def extract_from_weight(args, layer_id, original_weight, ones, zeros, is_robust):
    original_weight = original_weight.detach().cpu().numpy()
    chunk_length = args.chunk_length

    for row_id in range(original_weight.shape[0]):
        if not is_robust and not is_ellmark_watermark_row(args, layer_id, row_id):
            continue
        for col_id in range(original_weight.shape[1]):
            value = original_weight[row_id][col_id]
            if value == 0:
                continue
            use_weight, insert_position, bit_position = get_ellmark_bit_mapping(args, value, chunk_length)
            if not use_weight:
                continue
            inserted_bit = get_bit_from_weight(value, insert_position)
            if inserted_bit == "1":
                ones[bit_position] += 1
            else:
                zeros[bit_position] += 1

    return ones, zeros
