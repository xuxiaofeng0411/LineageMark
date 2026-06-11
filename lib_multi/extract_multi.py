import time
import torch
import numpy as np
import hashlib
import tqdm
from .utils_multi import (get_blocks,
                          find_layers,
                          string_to_binary,
                          get_bit_from_weight,
                          get_extract_acc,
                          qweight2weight,
                          is_watermark_row,
                          get_watermark_bit_mapping,
                          get_mean_diff_group_mapping,
                          get_projection_bit_mapping,
                          format_time
                          )

def extract_watermark(args, model, dssa_layer_masks=None):
    is_robust = args.mode == 'robust'
    print(f'mode: is_robust={is_robust}')
    layers = get_blocks(model)
    real_watermark = args.watermark  # watermark
    real_watermark_bits = string_to_binary(real_watermark)  # watermark bit string
    L = len(real_watermark_bits)  # the length of watermark bit string
    chunk_L = args.chunk_length
    chunk_num = L // chunk_L
    watermarks_from_layers = {'ones': [0] * L, 'zeors': [0] * L}
    gamma_layer = args.gamma_1
    layer_id = 0
    extract_layer_num = 0
    print(
        "Blind extraction enabled: expected watermark is used only for final "
        "accuracy, not for chunk matching, layer filtering, or early stop."
    )
    for i in tqdm.tqdm(range(len(layers)), desc="Running OurMark extract..."):
        # for i in tqdm.tqdm(range(1), desc="Running OurMark extract..."):
        layer = layers[i]
        # print(layer)
        subset = find_layers(layer)
        block_masks = (
            None if getattr(args, "data_independent_extract", False)
            else dssa_layer_masks.get(i, {}) if dssa_layer_masks is not None else None
        )
        for name in subset:
            ones = [0] * chunk_L
            zeros = [0] * chunk_L
            one_layer_time_start = time.time()
            print(f"[{layer_id}]: {name}: {subset[name]}")
            K_layer = args.password + str(layer_id)
            K_layer = int(hashlib.md5(K_layer.encode()).hexdigest(), 16)
            if K_layer % gamma_layer != 0:
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
            print(f"==> Extract watermark from weights...")
            if "8bit" in args.model:
                weight = qweight2weight(subset[name]).data.t()
            else:
                module = subset[name]
                if hasattr(module, 'qweight'):
                    from .utils_multi import qweight2weight
                    weight = qweight2weight(module).data.t()
                else:
                    weight = module.weight.data
            print(f"==> Extract watermark from weights...")
            module = subset[name]
            weight = module.weight.data
            original_weight = weight.cpu()
            ones, zeros = extract_from_weight(
                args, layer_id, original_weight, ones, zeros, is_robust, W_mask, name
            )
            print(f'ones: {ones}')
            print(f'zeros: {zeros}')
            extracted_chunk_watermark_bits = ''
            for i in range(chunk_L):
                if ones[i] > zeros[i]:
                    extracted_chunk_watermark_bits += '1'
                else:
                    extracted_chunk_watermark_bits += '0'
            chunk_id = extract_layer_num % chunk_num
            print(f'=========[{layer_id}]========[chunk {chunk_id}]')
            print(extracted_chunk_watermark_bits)
            one_layer_time_end = time.time()
            one_layer_time = one_layer_time_end - one_layer_time_start
            format_time(one_layer_time, "Extract of one layer")
            layer_id += 1
            extract_layer_num += 1
            for j in range(chunk_L):
                bit_idx = chunk_id * chunk_L + j
                watermarks_from_layers['ones'][bit_idx] += ones[j]
                watermarks_from_layers['zeors'][bit_idx] += zeros[j]
            extracted_watermark_bits = ''
            for j in range(L):
                if watermarks_from_layers['ones'][j] > watermarks_from_layers['zeors'][j]:
                    extracted_watermark_bits += '1'
                else:
                    extracted_watermark_bits += '0'
            cur_total_acc = get_extract_acc(real_watermark_bits, extracted_watermark_bits)
            print(f'Current ones: {watermarks_from_layers["ones"]}')
            print(f'Current zeros: {watermarks_from_layers["zeors"]}')
            print(f"Current final-only acc: {cur_total_acc}")
    torch.cuda.empty_cache()
    extracted_watermark_bits = ''
    for i in range(L):
        if watermarks_from_layers['ones'][i] > watermarks_from_layers['zeors'][i]:
            extracted_watermark_bits += '1'
        else:
            extracted_watermark_bits += '0'
    print(f"Real watermark: {real_watermark_bits}")
    print(f"Extract watermark: {extracted_watermark_bits}")
    return get_extract_acc(real_watermark_bits, extracted_watermark_bits)
def extract_from_weight(args, layer_id, original_weight, ones, zeros, is_robust,
                        W_mask=None, layer_name=""):
    wm_method = getattr(args, "wm_method", "projection")
    if wm_method == "projection":
        return extract_from_weight_projection(
            args, layer_id, original_weight, ones, zeros, is_robust, W_mask, layer_name
        )
    if wm_method == "mean_diff":
        return extract_from_weight_mean_diff(
            args, layer_id, original_weight, ones, zeros, is_robust, W_mask, layer_name
        )
    return extract_from_weight_bitflip(
        args, layer_id, original_weight, ones, zeros, is_robust, W_mask, layer_name
    )
def extract_from_weight_bitflip(args, layer_id, original_weight, ones, zeros, is_robust,
                                W_mask=None, layer_name=""):
    original_weight = original_weight.detach().cpu().numpy()
    chunk_length = args.chunk_length
    if W_mask is not None:
        if torch.is_tensor(W_mask):
            W_mask = W_mask.detach().cpu().numpy().astype(bool)
        else:
            W_mask = np.asarray(W_mask, dtype=bool)
    for i in range(original_weight.shape[0]):
        # Robust mode scans more layers, but rows and bit positions still use
        # the keyed v2 mapping so extraction matches insertion.
        if not is_watermark_row(args, layer_id, i, layer_name):
            continue
        for j in range(original_weight.shape[1]):
            if W_mask is not None and not W_mask[i][j]:
                continue
            if original_weight[i][j] == 0:
                continue
            use_weight, insert_position, insert_bit_position = get_watermark_bit_mapping(
                args, layer_id, i, j, chunk_length, layer_name
            )
            if not use_weight:
                continue
            inserted_bit = get_bit_from_weight(original_weight[i][j], insert_position)
            if inserted_bit == '1':
                ones[insert_bit_position] += 1
            else:
                zeros[insert_bit_position] += 1
    return ones, zeros
                                  
def extract_from_weight_mean_diff(args, layer_id, original_weight, ones, zeros, is_robust,
                                  W_mask=None, layer_name=""):
    original_weight = original_weight.detach().cpu().numpy()
    chunk_length = args.chunk_length
    use_dssa_mask = W_mask is not None
    if use_dssa_mask:
        if torch.is_tensor(W_mask):
            W_mask = W_mask.detach().cpu().numpy().astype(bool)
        else:
            W_mask = np.asarray(W_mask, dtype=bool)
    sums = [[0.0, 0.0] for _ in range(chunk_length)]
    counts = [[0, 0] for _ in range(chunk_length)]
    for i in range(original_weight.shape[0]):
        if not is_watermark_row(args, layer_id, i, layer_name):
            continue
        for j in range(original_weight.shape[1]):
            if use_dssa_mask and not W_mask[i][j]:
                continue
            if original_weight[i][j] == 0:
                continue
            use_weight, bit_pos, group_id = get_mean_diff_group_mapping(
                args, layer_id, i, j, chunk_length, layer_name
            )
            if not use_weight:
                continue
            sums[bit_pos][group_id] += float(original_weight[i][j])
            counts[bit_pos][group_id] += 1
    margins = []
    for bit_pos in range(chunk_length):
        if counts[bit_pos][0] == 0 or counts[bit_pos][1] == 0:
            margins.append(0.0)
            continue

        mean0 = sums[bit_pos][0] / counts[bit_pos][0]
        mean1 = sums[bit_pos][1] / counts[bit_pos][1]
        diff = mean1 - mean0
        vote_weight = abs(diff) * min(counts[bit_pos][0], counts[bit_pos][1])
        margins.append(diff)

        if diff > 0:
            ones[bit_pos] += vote_weight
        else:
            zeros[bit_pos] += vote_weight
    return ones, zeros

def extract_from_weight_projection(args, layer_id, original_weight, ones, zeros, is_robust,
                                   W_mask=None, layer_name=""):
    original_weight = original_weight.detach().cpu().numpy()
    chunk_length = args.chunk_length
    use_dssa_mask = W_mask is not None
    if use_dssa_mask:
        if torch.is_tensor(W_mask):
            W_mask = W_mask.detach().cpu().numpy().astype(bool)
        else:
            W_mask = np.asarray(W_mask, dtype=bool)
    projections = [0.0] * chunk_length
    counts = [0] * chunk_length

    for i in range(original_weight.shape[0]):
        if not is_watermark_row(args, layer_id, i, layer_name):
            continue
        for j in range(original_weight.shape[1]):
            if use_dssa_mask and not W_mask[i][j]:
                continue
            if original_weight[i][j] == 0:
                continue
            use_weight, bit_pos, coeff = get_projection_bit_mapping(
                args, layer_id, i, j, chunk_length, layer_name
            )
            if not use_weight:
                continue
            projections[bit_pos] += coeff * float(original_weight[i][j])
            counts[bit_pos] += 1
    for bit_pos in range(chunk_length):
        if counts[bit_pos] == 0:
            continue
        vote_weight = abs(projections[bit_pos])
        if projections[bit_pos] > 0:
            ones[bit_pos] += vote_weight
        else:
            zeros[bit_pos] += vote_weight
    print(f'projection counts: {counts}')
    print(f'projection values: {[round(x, 8) for x in projections]}')
    return ones, zeros
