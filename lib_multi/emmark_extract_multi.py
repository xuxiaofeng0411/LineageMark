import time

import numpy as np
import torch
import tqdm

from .utils_multi import (
    find_layers,
    format_time,
    get_blocks,
    get_extract_acc,
    qweight2weight,
    string_to_binary,
)


def _is_quantized_module(module):
    return hasattr(module, "qweight") and hasattr(module, "bits")


def _read_weight(module):
    if _is_quantized_module(module):
        return qweight2weight(module).data.cpu()
    return module.weight.data.cpu()


def extract_watermark(args, model, inserted_model, indices):
    real_watermark_bits = string_to_binary(args.watermark)
    real_watermark_length = len(real_watermark_bits)
    rng = np.random.default_rng(seed=args.seed)

    blocks = get_blocks(model)
    inserted_blocks = get_blocks(inserted_model)
    total_layer_num = sum(len(find_layers(block)) for block in blocks)

    every_layer_insert_bits_num = real_watermark_length // total_layer_num + 1
    modify_num = int(args.hidden_size * args.modify_rate)
    every_layer_modify_weight_num_per_bit = modify_num // every_layer_insert_bits_num
    every_layer_insert_candidate_bits_num = modify_num * args.candidate_rate
    print(f"every_layer_insert_bits_num: {every_layer_insert_bits_num}")

    extract_watermarks = [0] * real_watermark_length
    insert_bit_position = 0
    layer_id = 0

    for block_index in tqdm.tqdm(range(len(blocks)), desc="Running EmMark extract..."):
        linear_layers = find_layers(blocks[block_index])
        inserted_linear_layers = find_layers(inserted_blocks[block_index])

        for name, module in linear_layers.items():
            original_weight = _read_weight(module)
            inserted_weight = _read_weight(inserted_linear_layers[name])
            one_layer_start = time.time()
            print(f"[{layer_id}]: {name}: {module}")

            layer_indices = indices[layer_id].view(-1, 2).cpu()
            print("==> Extract watermark from weights...")

            for _ in range(every_layer_insert_bits_num):
                one = 0
                zero = 0
                for _ in range(every_layer_modify_weight_num_per_bit):
                    candidate_id = rng.integers(0, every_layer_insert_candidate_bits_num)
                    row_id, col_id = layer_indices[candidate_id]
                    row_id = int(row_id.item())
                    col_id = int(col_id.item())
                    delta = inserted_weight[row_id, col_id] - original_weight[row_id, col_id]
                    if delta == -1:
                        zero += 1
                    elif delta == 1:
                        one += 1
                extract_watermarks[insert_bit_position] = 1 if one > zero else 0
                insert_bit_position = (insert_bit_position + 1) % real_watermark_length

            format_time(time.time() - one_layer_start, "Extract of one layer")
            layer_id += 1

    torch.cuda.empty_cache()
    extracted_watermark_bits = "".join("1" if bit else "0" for bit in extract_watermarks)
    print(f"Real watermark: {real_watermark_bits}")
    print(f"Extract watermark: {extracted_watermark_bits}")
    return get_extract_acc(real_watermark_bits, extracted_watermark_bits)
