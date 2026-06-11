import time
import torch
import numpy as np
import hashlib
import tqdm

from .data_multi import get_loaders
from .layerwrapper_multi import WrappedGPT
from .utils_multi import (prepare_calibration_input,
                          get_blocks,
                          find_layers,
                          string_to_binary,
                          modify_bit_of_weight,
                          weight2qweight,
                          qweight2weight,
                          get_bit_from_weight,
                          is_watermark_row,
                          get_watermark_bit_mapping,
                          get_mean_diff_group_mapping,
                          get_projection_bit_mapping,
                          format_time
                          )

def insert_watermark(args, model, tokenizer, device, dataset_name='wikitext2', dssa_layer_masks=None):
    use_dssa = dssa_layer_masks is not None
    layers = get_blocks(model)
    if not use_dssa:
        print("Loading calibdation data...")
        dataloader, _ = get_loaders(dataset_name, nsamples=args.nsamples, seed=args.seed, seqlen=model.seqlen,
                                    tokenizer=tokenizer)

        with torch.no_grad():
            inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, args)
        inps.to(device)
        outs.to(device)
        attention_mask.to(device)
        if position_ids != None:
            position_ids.to(device)
    gamma_layer = args.gamma_1
    layer_id = 0
    insert_layer_num = 0
    change_weight_num = 0
    total_weight_num = 0
    total_preprocess_time = 0.0
    total_insert_time = 0.0
    for i in tqdm.tqdm(range(len(layers)), desc="Running OurMark insert..."):
        layer = layers[i]
        layer.to(device)
        subset = find_layers(layer)
        if not use_dssa:
            preprocess_time_start = time.time()
            wrapped_layers = {}
            for name in subset:
                if "8bit" in args.model:
                    wrapped_layers[name] = WrappedGPT(subset[name], layer_id, name, True)
                else:
                    wrapped_layers[name] = WrappedGPT(subset[name], layer_id, name)
            def add_batch(name):
                def tmp(_, inp, out):
                    wrapped_layers[name].add_batch(inp[0].data, out.data)
                return tmp
            handles = []
            for name in wrapped_layers:
                handles.append(subset[name].register_forward_hook(add_batch(name)))
            for j in range(args.nsamples):
                with torch.no_grad():
                    if position_ids != None:
                        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask, position_ids=position_ids)[
                            0]
                    else:
                        outs[j] = layer(inps[j].unsqueeze(0), attention_mask=attention_mask)[0]
            for h in handles:
                h.remove()
            preprocess_time_end = time.time()
            preprocess_time = preprocess_time_end - preprocess_time_start
            total_preprocess_time += preprocess_time
        for name in subset:
            print(f"[{layer_id}]: {name}: {subset[name]}")
            K_layer = args.password + str(layer_id)
            K_layer = int(hashlib.md5(K_layer.encode()).hexdigest(), 16)
            if K_layer % gamma_layer != 0:
                print(f"[{layer_id}]: This layer is skipped.")
                layer_id += 1
                continue
            # --- get weight matrix ---
            if "8bit" in args.model:
                weight = qweight2weight(subset[name]).data.t()
            else:
                weight = subset[name].weight.data
            original_weight = weight.cpu()
            # --- get W_mask ---
            if use_dssa:
                block_masks = dssa_layer_masks.get(i, {})
                if name not in block_masks:
                    print(f"    [{name}] not in DSSA masks, skipping.")
                    layer_id += 1
                    continue
                W_mask = block_masks[name]['W_mask']
            else:
                print(f'==> Get socres of weights...')
                preprocess_time_start = time.time()
                W_metric = torch.abs(original_weight) * torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1))).to(
                    original_weight.device)
                W_mask = (torch.zeros_like(W_metric) == 1)  ## initialize a mask to be all False
                sort_res = torch.sort(W_metric, dim=-1, stable=True)
                indices = sort_res[1][:, -int(W_metric.shape[1] * args.select_ratio):]
                W_mask.scatter_(1, indices, True)
                preprocess_time_end = time.time()
                preprocess_time = preprocess_time_end - preprocess_time_start
                total_preprocess_time += preprocess_time
            print(f"==> Insert watermark into weights...")
            insert_time_start = time.time()
            modified_weight = change_weight(
                args, layer_id, insert_layer_num, original_weight, W_mask, name
            )
            modified_weight_tensor = torch.tensor(modified_weight, dtype=original_weight.dtype)
            if "8bit" in args.model:
                modified_qweight = weight2qweight(modified_weight_tensor.t(), subset[name])
                subset[name].qweight.data = modified_qweight
            else:
                subset[name].weight.data = modified_weight_tensor
            # modify result
            difference = modified_weight_tensor - weight.to(modified_weight_tensor.device)
            non_zero_count = torch.count_nonzero(difference)
            weight_count = original_weight.shape[0] * original_weight.shape[1]
            print(f'==> Modify num: {non_zero_count}\tratio:{non_zero_count / weight_count}')
            insert_time_end = time.time()
            insert_time = insert_time_end - insert_time_start
            total_insert_time += insert_time
            change_weight_num += non_zero_count
            total_weight_num += weight_count
            layer_id += 1
            insert_layer_num += 1
        if not use_dssa:
            inps, outs = outs, inps
    format_time(total_preprocess_time, 'Preprocess')
    format_time(total_insert_time, 'Insert')
    torch.cuda.empty_cache()
    return change_weight_num, total_weight_num

def change_weight(args, layer_id, insert_layer_num, original_weight, W_mask, layer_name=""):
    wm_method = getattr(args, "wm_method", "projection")
    if wm_method == "projection":
        return change_weight_projection(args, layer_id, insert_layer_num, original_weight, W_mask, layer_name)
    if wm_method == "mean_diff":
        return change_weight_mean_diff(args, layer_id, insert_layer_num, original_weight, W_mask, layer_name)
    return change_weight_bitflip(args, layer_id, insert_layer_num, original_weight, W_mask, layer_name)

def _as_bool_mask(W_mask):
    if W_mask is None:
        return None
    if torch.is_tensor(W_mask):
        return W_mask.detach().cpu().numpy().astype(bool)
    return np.asarray(W_mask, dtype=bool)

def change_weight_bitflip(args, layer_id, insert_layer_num, original_weight, W_mask, layer_name=""):
    # print(f'[insert layer num]: {insert_layer_num}')
    original_weight = original_weight.detach().cpu().numpy()
    W = args.watermark  # watermark="mark"
    chunk_id = insert_layer_num % len(W)
    chunk = W[chunk_id]
    W = string_to_binary(chunk)  # watermark bit string
    print(f'=========[{layer_id}]========[{W}]')
    L = len(W)  # the length of watermark bit string
    data_independent = getattr(args, "data_independent_extract", False)
    edit_mask = _as_bool_mask(W_mask)
    detect_mask = None if data_independent else edit_mask
    # 1. get the one-zero distribution of positions to be inserted
    zeros = [0] * L
    ones = [0] * L
    for i in range(original_weight.shape[0]):
        if not is_watermark_row(args, layer_id, i, layer_name):
            continue
        for j in range(original_weight.shape[1]):
            if detect_mask is not None and not detect_mask[i][j]:
                continue
            if original_weight[i][j] == 0:
                continue
            use_weight, insert_position, insert_bit_position = get_watermark_bit_mapping(
                args, layer_id, i, j, L, layer_name
            )
            if not use_weight:
                continue
            original_weight_bit = get_bit_from_weight(original_weight[i][j], insert_position)
            if original_weight_bit == '1':
                ones[insert_bit_position] += 1
            else:
                zeros[insert_bit_position] += 1
    print(f'current distribution: ')
    print(f'ones: {ones}')
    print(f'zeros: {zeros}')

    # 2. calculate reverse bits num
    reverse_bits_list = [0] * L
    for _ in range(L):
        difference = abs(ones[_] - zeros[_]) // 2
        if W[_] == "0":
            if ones[_] > zeros[_]:
                reverse_bits_list[_] += difference
                delta = (zeros[_] + difference) // 10
                delta = max(delta, args.delta)
                reverse_bits_list[_] += delta
            else:
                more = zeros[_] - ones[_]
                delta = (ones[_]) // 10
                delta = max(delta, args.delta)
                if more < delta:
                    reverse_bits_list[_] += delta - more
        else:
            if ones[_] < zeros[_]:
                reverse_bits_list[_] += difference
                delta = (ones[_] + difference) // 10
                delta = max(delta, args.delta)
                reverse_bits_list[_] += delta
            else:
                more = ones[_] - zeros[_]
                delta = (zeros[_]) // 10
                delta = max(delta, args.delta)
                if more < delta:
                    reverse_bits_list[_] += delta - more
    print(reverse_bits_list)
    # 3. begin reverse
    for i in range(original_weight.shape[0]):
        if not is_watermark_row(args, layer_id, i, layer_name):
            continue
        for j in range(original_weight.shape[1]):
            if edit_mask is not None and not edit_mask[i][j]:
                continue
            if original_weight[i][j] == 0:
                continue
            use_weight, insert_position, insert_bit_position = get_watermark_bit_mapping(
                args, layer_id, i, j, L, layer_name
            )
            if not use_weight:
                continue
            if sum(reverse_bits_list) == 0:
                return original_weight
            if reverse_bits_list[insert_bit_position] == 0:
                continue
            insert_bit = W[insert_bit_position]
            # print(f'input weight: {original_weight[i][j]}')
            original_weight_bit = get_bit_from_weight(original_weight[i][j], insert_position)
            if original_weight_bit == insert_bit:
                continue
            original_weight[i][j] = modify_bit_of_weight(original_weight[i][j], insert_bit, insert_position)
            reverse_bits_list[insert_bit_position] -= 1
            # print(f'output weight: {original_weight[i][j]}')
    return original_weight

def change_weight_mean_diff(args, layer_id, insert_layer_num, original_weight, W_mask, layer_name=""):
    original_weight = original_weight.detach().cpu().numpy()
    W = args.watermark
    chunk_id = insert_layer_num % len(W)
    chunk = W[chunk_id]
    W = string_to_binary(chunk)
    print(f'=========[{layer_id}]========[{W}]')
    L = len(W)
    data_independent = getattr(args, "data_independent_extract", False)
    edit_mask = _as_bool_mask(W_mask)
    detect_mask = None if data_independent else edit_mask
    detect_groups = [[[] for _ in range(2)] for _ in range(L)]
    edit_groups = [[[] for _ in range(2)] for _ in range(L)]
    for i in range(original_weight.shape[0]):
        if not is_watermark_row(args, layer_id, i, layer_name):
            continue
        for j in range(original_weight.shape[1]):
            if detect_mask is not None and not detect_mask[i][j]:
                continue
            if original_weight[i][j] == 0:
                continue
            use_weight, bit_pos, group_id = get_mean_diff_group_mapping(
                args, layer_id, i, j, L, layer_name
            )
            if not use_weight:
                continue
            detect_groups[bit_pos][group_id].append((i, j))
            if edit_mask is None or edit_mask[i][j]:
                edit_groups[bit_pos][group_id].append((i, j))
    margin = float(getattr(args, "mean_margin", 0.02))
    detect_counts = [(len(detect_groups[i][0]), len(detect_groups[i][1])) for i in range(L)]
    edit_counts = [(len(edit_groups[i][0]), len(edit_groups[i][1])) for i in range(L)]
    print(f'mean-diff detect group counts: {detect_counts}')
    print(f'mean-diff editable group counts: {edit_counts}')
    changed = 0
    for bit_pos in range(L):
        detect_g0 = detect_groups[bit_pos][0]
        detect_g1 = detect_groups[bit_pos][1]
        edit_g0 = edit_groups[bit_pos][0]
        edit_g1 = edit_groups[bit_pos][1]
        if len(detect_g0) == 0 or len(detect_g1) == 0:
            print(f'bit {bit_pos}: skipped, empty detect group g0={len(detect_g0)} g1={len(detect_g1)}')
            continue
        if len(edit_g0) == 0 or len(edit_g1) == 0:
            print(f'bit {bit_pos}: skipped, empty editable group g0={len(edit_g0)} g1={len(edit_g1)}')
            continue
        mean0 = float(np.mean([original_weight[i][j] for i, j in detect_g0]))
        mean1 = float(np.mean([original_weight[i][j] for i, j in detect_g1]))
        diff = mean1 - mean0
        target_sign = 1.0 if W[bit_pos] == "1" else -1.0
        signed_margin = target_sign * diff
        if signed_margin >= margin:
            print(
                f'bit {bit_pos}: keep bit={W[bit_pos]} '
                f'mean0={mean0:.8f} mean1={mean1:.8f} diff={diff:.8f}'
            )
            continue
        gap = margin - signed_margin
        editable_ratio = len(edit_g1) / len(detect_g1) + len(edit_g0) / len(detect_g0)
        if editable_ratio <= 0:
            print(f'bit {bit_pos}: skipped, no editable effect on detector groups')
            continue
        shift = np.array((gap / editable_ratio) * target_sign, dtype=original_weight.dtype).item()
        for i, j in edit_g1:
            original_weight[i][j] = np.array(original_weight[i][j] + shift, dtype=original_weight.dtype)
            changed += 1
        for i, j in edit_g0:
            original_weight[i][j] = np.array(original_weight[i][j] - shift, dtype=original_weight.dtype)
            changed += 1
        new_mean0 = float(np.mean([original_weight[i][j] for i, j in detect_g0]))
        new_mean1 = float(np.mean([original_weight[i][j] for i, j in detect_g1]))
        print(
            f'bit {bit_pos}: embed bit={W[bit_pos]} '
            f'before_diff={diff:.8f} after_diff={new_mean1 - new_mean0:.8f} '
            f'changed={len(edit_g0) + len(edit_g1)}'
        )
    print(f'mean-diff changed coordinate visits: {changed}')
    return original_weight

def change_weight_projection(args, layer_id, insert_layer_num, original_weight, W_mask, layer_name=""):
    original_weight = original_weight.detach().cpu().numpy()
    W = args.watermark
    chunk_id = insert_layer_num % len(W)
    chunk = W[chunk_id]
    W = string_to_binary(chunk)
    print(f'=========[{layer_id}]========[{W}]')
    L = len(W)
    data_independent = getattr(args, "data_independent_extract", False)
    edit_mask = _as_bool_mask(W_mask)
    detect_mask = None if data_independent else edit_mask
    detect_groups = [[] for _ in range(L)]
    edit_groups = [[] for _ in range(L)]
    for i in range(original_weight.shape[0]):
        if not is_watermark_row(args, layer_id, i, layer_name):
            continue
        for j in range(original_weight.shape[1]):
            if detect_mask is not None and not detect_mask[i][j]:
                continue
            if original_weight[i][j] == 0:
                continue
            use_weight, bit_pos, coeff = get_projection_bit_mapping(
                args, layer_id, i, j, L, layer_name
            )
            if not use_weight:
                continue
            detect_groups[bit_pos].append((i, j, coeff))
            if edit_mask is None or edit_mask[i][j]:
                edit_groups[bit_pos].append((i, j, coeff))
    margin = float(getattr(args, "projection_margin", 0.5))
    max_update = float(getattr(args, "projection_max_update", 0.0))
    detect_counts = [len(detect_groups[i]) for i in range(L)]
    edit_counts = [len(edit_groups[i]) for i in range(L)]
    print(f'projection detect group counts: {detect_counts}')
    print(f'projection editable group counts: {edit_counts}')
    changed = 0
    for bit_pos in range(L):
        detect_coords = detect_groups[bit_pos]
        edit_coords = edit_groups[bit_pos]
        n_edit = len(edit_coords)
        if len(detect_coords) == 0:
            print(f'bit {bit_pos}: skipped, empty projection detect group')
            continue
        if n_edit == 0:
            print(f'bit {bit_pos}: skipped, empty projection editable group')
            continue
        projection = float(sum(coeff * float(original_weight[i][j]) for i, j, coeff in detect_coords))
        target_sign = 1.0 if W[bit_pos] == "1" else -1.0
        signed_margin = target_sign * projection

        if signed_margin >= margin:
            print(
                f'bit {bit_pos}: keep bit={W[bit_pos]} '
                f'projection={projection:.8f} signed_margin={signed_margin:.8f}'
            )
            continue
        gap = margin - signed_margin
        step = gap / float(n_edit)
        if max_update > 0:
            step = min(step, max_update)
        update = np.array(target_sign * step, dtype=original_weight.dtype).item()
        for i, j, coeff in edit_coords:
            original_weight[i][j] = np.array(
                original_weight[i][j] + coeff * update,
                dtype=original_weight.dtype,
            )
            changed += 1
        new_projection = float(sum(coeff * float(original_weight[i][j]) for i, j, coeff in detect_coords))
        print(
            f'bit {bit_pos}: embed bit={W[bit_pos]} '
            f'before_projection={projection:.8f} after_projection={new_projection:.8f} '
            f'step={step:.8f} changed={n_edit}'
        )
    print(f'projection changed coordinate visits: {changed}')
    return original_weight



