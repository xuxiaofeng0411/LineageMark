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
    get_extract_acc,
    get_mean_diff_group_mapping,
    get_projection_bit_mapping,
    get_watermark_bit_mapping,
    is_watermark_row,
    qweight2weight,
    string_to_binary,
)


def _layer_selected(args, layer_id):
    layer_key = args.password + str(layer_id)
    layer_key = int(hashlib.md5(layer_key.encode()).hexdigest(), 16)
    return layer_key % args.gamma_1 == 0


def _block_masks_for(args, dssa_layer_masks, block_index):
    if getattr(args, "data_independent_extract", False):
        return None
    if dssa_layer_masks is None:
        return None
    return dssa_layer_masks.get(block_index, {})


def _mask_to_bool_array(W_mask):
    if W_mask is None:
        return None
    if torch.is_tensor(W_mask):
        return W_mask.detach().cpu().numpy().astype(bool)
    return np.asarray(W_mask, dtype=bool)


def _empty_vote_buffers(length):
    return [0] * length, [0] * length


def _bits_from_votes(ones, zeros):
    return ''.join('1' if ones[idx] > zeros[idx] else '0' for idx in range(len(ones)))


def _merge_chunk_votes(target_votes, chunk_id, ones, zeros):
    chunk_length = len(ones)
    for offset in range(chunk_length):
        bit_idx = chunk_id * chunk_length + offset
        target_votes['ones'][bit_idx] += ones[offset]
        target_votes['zeors'][bit_idx] += zeros[offset]


def _iter_watermark_coordinates(args, layer_id, weights, W_mask, layer_name):
    mask = _mask_to_bool_array(W_mask)
    for row_id, col_id in np.ndindex(weights.shape):
        if not is_watermark_row(args, layer_id, row_id, layer_name):
            continue
        if mask is not None and not mask[row_id][col_id]:
            continue
        if weights[row_id][col_id] == 0:
            continue
        yield row_id, col_id


def _read_qweight(module, unpacker):
    return unpacker(module).data.t()


def _manual_dequantize(module):
    weight_fp = (module.qweight * module.scales - module.qzeros).to(torch.float16)
    return weight_fp.data.t()


def _extract_weight_like_original(args, module):
    print(f"==> Extract watermark from weights...")

    if "8bit" in args.model:
        weight = _read_qweight(module, qweight2weight)
    else:
        if hasattr(module, 'qweight'):
            from .utils_multi import qweight2weight
            weight = _read_qweight(module, qweight2weight)
        else:
            weight = module.weight.data

    print(f"==> Extract watermark from weights...")

    if "8bit" in args.model:
        weight = _read_qweight(module, qweight2weight)
    elif hasattr(module, 'qweight'):
        if callable(qweight2weight):
            weight = _read_qweight(module, qweight2weight)
        else:
            weight = _manual_dequantize(module)
    else:
        weight = module.weight.data

    return weight.cpu()


def extract_watermark(args, model, dssa_layer_masks=None):
    is_robust = args.mode == 'robust'
    print(f'mode: is_robust={is_robust}')

    layers = get_blocks(model)
    real_watermark = args.watermark
    real_watermark_bits = string_to_binary(real_watermark)
    L = len(real_watermark_bits)
    chunk_L = args.chunk_length
    chunk_num = L // chunk_L
    watermarks_from_layers = {'ones': [0] * L, 'zeors': [0] * L}

    layer_id = 0
    extract_layer_num = 0
    print(
        "Blind extraction enabled: expected watermark is used only for final "
        "accuracy, not for chunk matching, layer filtering, or early stop."
    )

    iterator = tqdm.tqdm(enumerate(layers), total=len(layers), desc="Running OurMark extract...")
    for block_index, layer in iterator:
        subset = find_layers(layer)
        block_masks = _block_masks_for(args, dssa_layer_masks, block_index)

        for name, module in subset.items():
            ones, zeros = _empty_vote_buffers(chunk_L)
            one_layer_time_start = time.time()
            print(f"[{layer_id}]: {name}: {module}")

            if not _layer_selected(args, layer_id):
                print(f"[{layer_id}]: This layer is skipped.")
                layer_id += 1
                continue

            W_mask = None
            if block_masks is not None:
                if name not in block_masks:
                    print(f"    [{name}] not in saved watermark map, skipping.")
                    layer_id += 1
                    continue
                W_mask = block_masks[name]['W_mask']

            original_weight = _extract_weight_like_original(args, module)
            ones, zeros = extract_from_weight(
                args, layer_id, original_weight, ones, zeros, is_robust, W_mask, name
            )
            print(f'ones: {ones}')
            print(f'zeros: {zeros}')

            extracted_chunk_watermark_bits = _bits_from_votes(ones, zeros)
            chunk_id = extract_layer_num % chunk_num
            print(f'=========[{layer_id}]========[chunk {chunk_id}]')
            print(extracted_chunk_watermark_bits)

            one_layer_time = time.time() - one_layer_time_start
            format_time(one_layer_time, "Extract of one layer")

            layer_id += 1
            extract_layer_num += 1
            _merge_chunk_votes(watermarks_from_layers, chunk_id, ones, zeros)

            extracted_watermark_bits = _bits_from_votes(
                watermarks_from_layers['ones'], watermarks_from_layers['zeors']
            )
            cur_total_acc = get_extract_acc(real_watermark_bits, extracted_watermark_bits)
            print(f'Current ones: {watermarks_from_layers["ones"]}')
            print(f'Current zeros: {watermarks_from_layers["zeors"]}')
            print(f"Current final-only acc: {cur_total_acc}")

    torch.cuda.empty_cache()
    extracted_watermark_bits = _bits_from_votes(
        watermarks_from_layers['ones'], watermarks_from_layers['zeors']
    )

    print(f"Real watermark: {real_watermark_bits}")
    print(f"Extract watermark: {extracted_watermark_bits}")
    return get_extract_acc(real_watermark_bits, extracted_watermark_bits)


def extract_from_weight(args, layer_id, original_weight, ones, zeros, is_robust,
                        W_mask=None, layer_name=""):
    method = getattr(args, "wm_method", "projection")
    method_impl = {
        "projection": extract_from_weight_projection,
        "mean_diff": extract_from_weight_mean_diff,
    }.get(method, extract_from_weight_bitflip)
    return method_impl(args, layer_id, original_weight, ones, zeros, is_robust, W_mask, layer_name)


def extract_from_weight_bitflip(args, layer_id, original_weight, ones, zeros, is_robust,
                                W_mask=None, layer_name=""):
    original_weight = original_weight.detach().cpu().numpy()
    chunk_length = args.chunk_length

    for row_id, col_id in _iter_watermark_coordinates(args, layer_id, original_weight, W_mask, layer_name):
        use_weight, insert_position, insert_bit_position = get_watermark_bit_mapping(
            args, layer_id, row_id, col_id, chunk_length, layer_name
        )
        if not use_weight:
            continue

        inserted_bit = get_bit_from_weight(original_weight[row_id][col_id], insert_position)
        if inserted_bit == '1':
            ones[insert_bit_position] += 1
        else:
            zeros[insert_bit_position] += 1

    return ones, zeros


def extract_from_weight_mean_diff(args, layer_id, original_weight, ones, zeros, is_robust,
                                  W_mask=None, layer_name=""):
    original_weight = original_weight.detach().cpu().numpy()
    chunk_length = args.chunk_length
    sums = [[0.0, 0.0] for _ in range(chunk_length)]
    counts = [[0, 0] for _ in range(chunk_length)]

    for row_id, col_id in _iter_watermark_coordinates(args, layer_id, original_weight, W_mask, layer_name):
        use_weight, bit_pos, group_id = get_mean_diff_group_mapping(
            args, layer_id, row_id, col_id, chunk_length, layer_name
        )
        if not use_weight:
            continue

        sums[bit_pos][group_id] += float(original_weight[row_id][col_id])
        counts[bit_pos][group_id] += 1

    margins = []
    for bit_pos, bit_counts in enumerate(counts):
        if bit_counts[0] == 0 or bit_counts[1] == 0:
            margins.append(0.0)
            continue

        mean0 = sums[bit_pos][0] / bit_counts[0]
        mean1 = sums[bit_pos][1] / bit_counts[1]
        diff = mean1 - mean0
        vote_weight = abs(diff) * min(bit_counts[0], bit_counts[1])
        margins.append(diff)

        if diff > 0:
            ones[bit_pos] += vote_weight
        else:
            zeros[bit_pos] += vote_weight

    print(f'mean-diff counts: {counts}')
    print(f'mean-diff margins: {[round(x, 8) for x in margins]}')
    return ones, zeros


def extract_from_weight_projection(args, layer_id, original_weight, ones, zeros, is_robust,
                                   W_mask=None, layer_name=""):
    original_weight = original_weight.detach().cpu().numpy()
    chunk_length = args.chunk_length
    projections = [0.0] * chunk_length
    counts = [0] * chunk_length

    for row_id, col_id in _iter_watermark_coordinates(args, layer_id, original_weight, W_mask, layer_name):
        use_weight, bit_pos, coeff = get_projection_bit_mapping(
            args, layer_id, row_id, col_id, chunk_length, layer_name
        )
        if not use_weight:
            continue

        projections[bit_pos] += coeff * float(original_weight[row_id][col_id])
        counts[bit_pos] += 1

    for bit_pos, projection in enumerate(projections):
        if counts[bit_pos] == 0:
            continue

        vote_weight = abs(projection)
        if projection > 0:
            ones[bit_pos] += vote_weight
        else:
            zeros[bit_pos] += vote_weight

    print(f'projection counts: {counts}')
    print(f'projection values: {[round(x, 8) for x in projections]}')
    return ones, zeros
