import hashlib
import time
import numpy as np
import torch
import tqdm
from .data_multi import get_loaders
from .layerwrapper_multi import WrappedGPT
from .utils_multi import (
    prepare_calibration_input,
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
    format_time,
)

def insert_watermark(args, model, tokenizer, device, dataset_name='wikitext2', dssa_layer_masks=None):
    """
    Insert a keyed watermark into model weights.
    If DSSA masks are supplied, the insertion pass reuses those masks directly.
    Otherwise it computes the activation-weight mask in the same way as the
    original insertion path.
    """
    using_dssa_masks = dssa_layer_masks is not None
    blocks = get_blocks(model)
    if not using_dssa_masks:
        print("Loading calibdation data...")
        dataloader, _ = get_loaders(
            dataset_name,
            nsamples=args.nsamples,
            seed=args.seed,
            seqlen=model.seqlen,
            tokenizer=tokenizer,
        )
        with torch.no_grad():
            inps, outs, attention_mask, position_ids = prepare_calibration_input(
                model, dataloader, device, args
            )
        inps.to(device)
        outs.to(device)
        attention_mask.to(device)
        if position_ids is not None:
            position_ids.to(device)
    layer_gate = args.gamma_1
    layer_id = 0
    insert_layer_num = 0
    change_weight_num = 0
    total_weight_num = 0
    total_preprocess_time = 0.0
    total_insert_time = 0.0
    iterator = tqdm.tqdm(enumerate(blocks), total=len(blocks), desc="Running OurMark insert...")
    for block_index, block in iterator:
        block.to(device)
        linear_layers = find_layers(block)
        wrapped_layers = {}
        if not using_dssa_masks:
            preprocess_start = time.time()
            wrapped_layers = {
                name: WrappedGPT(module, layer_id, name, "8bit" in args.model)
                for name, module in linear_layers.items()
            }
            handles = []
            for hook_name, wrapped in wrapped_layers.items():
                handles.append(
                    linear_layers[hook_name].register_forward_hook(
                        lambda _module, inp, out, wrapped=wrapped: wrapped.add_batch(inp[0].data, out.data)
                    )
                )
            try:
                for sample_id in range(args.nsamples):
                    with torch.no_grad():
                        block_input = inps[sample_id].unsqueeze(0)
                        if position_ids is None:
                            outs[sample_id] = block(block_input, attention_mask=attention_mask)[0]
                        else:
                            outs[sample_id] = block(
                                block_input,
                                attention_mask=attention_mask,
                                position_ids=position_ids,
                            )[0]
            finally:
                for handle in handles:
                    handle.remove()
            total_preprocess_time += time.time() - preprocess_start
        for name, module in linear_layers.items():
            print(f"[{layer_id}]: {name}: {module}")
            layer_key = int(hashlib.md5((args.password + str(layer_id)).encode()).hexdigest(), 16)
            if layer_key % layer_gate != 0:
                print(f"[{layer_id}]: This layer is skipped.")
                layer_id += 1
                continue
            if "8bit" in args.model:
                weight = qweight2weight(module).data.t()
            else:
                weight = module.weight.data
            original_weight = weight.cpu()
            if using_dssa_masks:
                block_masks = dssa_layer_masks.get(block_index, {})
                if name not in block_masks:
                    print(f"    [{name}] not in DSSA masks, skipping.")
                    layer_id += 1
                    continue
                W_mask = block_masks[name]['W_mask']
            else:
                print(f'==> Get socres of weights...')
                preprocess_start = time.time()
                row_scale = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1))).to(original_weight.device)
                W_metric = torch.abs(original_weight) * row_scale
                W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
                n_columns = int(W_metric.shape[1] * args.select_ratio)
                chosen_columns = torch.sort(W_metric, dim=-1, stable=True)[1][:, -n_columns:]
                W_mask.scatter_(1, chosen_columns, True)
                total_preprocess_time += time.time() - preprocess_start
            print(f"==> Insert watermark into weights...")
            insert_start = time.time()
            modified_weight = change_weight(args, layer_id, insert_layer_num, original_weight, W_mask, name)
            modified_weight_tensor = torch.tensor(modified_weight, dtype=original_weight.dtype)
            if "8bit" in args.model:
                module.qweight.data = weight2qweight(modified_weight_tensor.t(), module)
            else:
                module.weight.data = modified_weight_tensor
            difference = modified_weight_tensor - weight.to(modified_weight_tensor.device)
            non_zero_count = torch.count_nonzero(difference)
            weight_count = original_weight.shape[0] * original_weight.shape[1]
            print(f'==> Modify num: {non_zero_count}\tratio:{non_zero_count / weight_count}')
            total_insert_time += time.time() - insert_start
            change_weight_num += non_zero_count
            total_weight_num += weight_count
            layer_id += 1
            insert_layer_num += 1
        if not using_dssa_masks:
            inps, outs = outs, inps
    format_time(total_preprocess_time, 'Preprocess')
    format_time(total_insert_time, 'Insert')
    torch.cuda.empty_cache()
    return change_weight_num, total_weight_num

def change_weight(args, layer_id, insert_layer_num, original_weight, W_mask, layer_name=""):
    method = getattr(args, "wm_method", "projection")
    method_impl = {
        "projection": change_weight_projection,
        "mean_diff": change_weight_mean_diff,
        "bitflip": change_weight_bitflip,
    }.get(method, change_weight_bitflip)
    return method_impl(args, layer_id, insert_layer_num, original_weight, W_mask, layer_name)

def _as_bool_mask(W_mask):
    if W_mask is None:
        return None
    return (
        W_mask.detach().cpu().numpy().astype(bool)
        if torch.is_tensor(W_mask)
        else np.asarray(W_mask, dtype=bool)
    )

def change_weight_bitflip(args, layer_id, insert_layer_num, original_weight, W_mask, layer_name=""):
    original_weight = original_weight.detach().cpu().numpy()
    watermark = args.watermark
    chunk = watermark[insert_layer_num % len(watermark)]
    watermark_bits = string_to_binary(chunk)
    print(f'=========[{layer_id}]========[{watermark_bits}]')
    bit_count = len(watermark_bits)
    edit_mask = _as_bool_mask(W_mask)
    detect_mask = None if getattr(args, "data_independent_extract", False) else edit_mask
    zeros = [0] * bit_count
    ones = [0] * bit_count
    for row_id, col_id in np.ndindex(original_weight.shape):
        if not is_watermark_row(args, layer_id, row_id, layer_name):
            continue
        if detect_mask is not None and not detect_mask[row_id][col_id]:
            continue
        value = original_weight[row_id][col_id]
        if value == 0:
            continue
        use_weight, insert_position, bit_position = get_watermark_bit_mapping(
            args, layer_id, row_id, col_id, bit_count, layer_name
        )
        if not use_weight:
            continue

        if get_bit_from_weight(value, insert_position) == '1':
            ones[bit_position] += 1
        else:
            zeros[bit_position] += 1
    print(f'current distribution: ')
    print(f'ones: {ones}')
    print(f'zeros: {zeros}')
    reverse_bits_list = [0] * bit_count
    for bit_idx, target_bit in enumerate(watermark_bits):
        desired = ones[bit_idx] if target_bit == "1" else zeros[bit_idx]
        opposite = zeros[bit_idx] if target_bit == "1" else ones[bit_idx]
        if desired < opposite:
            half_gap = (opposite - desired) // 2
            reverse_bits_list[bit_idx] = half_gap + max((desired + half_gap) // 10, args.delta)
            continue
        surplus = desired - opposite
        required_margin = max(opposite // 10, args.delta)
        if surplus < required_margin:
            reverse_bits_list[bit_idx] = required_margin - surplus
    print(reverse_bits_list)
    remaining = sum(reverse_bits_list)
    for row_id, col_id in np.ndindex(original_weight.shape):
        if not is_watermark_row(args, layer_id, row_id, layer_name):
            continue
        if edit_mask is not None and not edit_mask[row_id][col_id]:
            continue
        value = original_weight[row_id][col_id]
        if value == 0:
            continue
        use_weight, insert_position, bit_position = get_watermark_bit_mapping(
            args, layer_id, row_id, col_id, bit_count, layer_name
        )
        if not use_weight:
            continue
        if remaining == 0:
            return original_weight
        if reverse_bits_list[bit_position] == 0:
            continue
        target_bit = watermark_bits[bit_position]
        if get_bit_from_weight(value, insert_position) == target_bit:
            continue
        original_weight[row_id][col_id] = modify_bit_of_weight(value, target_bit, insert_position)
        reverse_bits_list[bit_position] -= 1
        remaining -= 1
    return original_weight

def change_weight_mean_diff(args, layer_id, insert_layer_num, original_weight, W_mask, layer_name=""):
    original_weight = original_weight.detach().cpu().numpy()
    watermark = args.watermark
    watermark_bits = string_to_binary(watermark[insert_layer_num % len(watermark)])
    print(f'=========[{layer_id}]========[{watermark_bits}]')
    bit_count = len(watermark_bits)
    edit_mask = _as_bool_mask(W_mask)
    detect_mask = None if getattr(args, "data_independent_extract", False) else edit_mask
    detect_groups = [[[] for _ in range(2)] for _ in range(bit_count)]
    edit_groups = [[[] for _ in range(2)] for _ in range(bit_count)]
    for row_id, col_id in np.ndindex(original_weight.shape):
        if not is_watermark_row(args, layer_id, row_id, layer_name):
            continue
        if detect_mask is not None and not detect_mask[row_id][col_id]:
            continue
        if original_weight[row_id][col_id] == 0:
            continue
        use_weight, bit_position, group_id = get_mean_diff_group_mapping(
            args, layer_id, row_id, col_id, bit_count, layer_name
        )
        if not use_weight:
            continue
        detect_groups[bit_position][group_id].append((row_id, col_id))
        if edit_mask is None or edit_mask[row_id][col_id]:
            edit_groups[bit_position][group_id].append((row_id, col_id))
    margin = float(getattr(args, "mean_margin", 0.02))
    detect_counts = [(len(groups[0]), len(groups[1])) for groups in detect_groups]
    edit_counts = [(len(groups[0]), len(groups[1])) for groups in edit_groups]
    print(f'mean-diff detect group counts: {detect_counts}')
    print(f'mean-diff editable group counts: {edit_counts}')
    changed = 0
    for bit_position, target_bit in enumerate(watermark_bits):
        detect_g0, detect_g1 = detect_groups[bit_position]
        edit_g0, edit_g1 = edit_groups[bit_position]
        if len(detect_g0) == 0 or len(detect_g1) == 0:
            print(f'bit {bit_position}: skipped, empty detect group g0={len(detect_g0)} g1={len(detect_g1)}')
            continue
        if len(edit_g0) == 0 or len(edit_g1) == 0:
            print(f'bit {bit_position}: skipped, empty editable group g0={len(edit_g0)} g1={len(edit_g1)}')
            continue
        mean0 = float(np.mean([original_weight[row_id][col_id] for row_id, col_id in detect_g0]))
        mean1 = float(np.mean([original_weight[row_id][col_id] for row_id, col_id in detect_g1]))
        diff = mean1 - mean0
        target_sign = 1.0 if target_bit == "1" else -1.0
        signed_margin = target_sign * diff
        if signed_margin >= margin:
            print(
                f'bit {bit_position}: keep bit={target_bit} '
                f'mean0={mean0:.8f} mean1={mean1:.8f} diff={diff:.8f}'
            )
            continue
        gap = margin - signed_margin
        editable_ratio = len(edit_g1) / len(detect_g1) + len(edit_g0) / len(detect_g0)
        if editable_ratio <= 0:
            print(f'bit {bit_position}: skipped, no editable effect on detector groups')
            continue
        shift = np.array((gap / editable_ratio) * target_sign, dtype=original_weight.dtype).item()
        for row_id, col_id in edit_g1:
            original_weight[row_id][col_id] = np.array(original_weight[row_id][col_id] + shift, dtype=original_weight.dtype)
            changed += 1
        for row_id, col_id in edit_g0:
            original_weight[row_id][col_id] = np.array(original_weight[row_id][col_id] - shift, dtype=original_weight.dtype)
            changed += 1
        new_mean0 = float(np.mean([original_weight[row_id][col_id] for row_id, col_id in detect_g0]))
        new_mean1 = float(np.mean([original_weight[row_id][col_id] for row_id, col_id in detect_g1]))
        print(
            f'bit {bit_position}: embed bit={target_bit} '
            f'before_diff={diff:.8f} after_diff={new_mean1 - new_mean0:.8f} '
            f'changed={len(edit_g0) + len(edit_g1)}'
        )
    print(f'mean-diff changed coordinate visits: {changed}')
    return original_weight

def change_weight_projection(args, layer_id, insert_layer_num, original_weight, W_mask, layer_name=""):
    original_weight = original_weight.detach().cpu().numpy()
    watermark = args.watermark
    watermark_bits = string_to_binary(watermark[insert_layer_num % len(watermark)])
    print(f'=========[{layer_id}]========[{watermark_bits}]')
    bit_count = len(watermark_bits)
    edit_mask = _as_bool_mask(W_mask)
    detect_mask = None if getattr(args, "data_independent_extract", False) else edit_mask
    detect_groups = [[] for _ in range(bit_count)]
    edit_groups = [[] for _ in range(bit_count)]
    for row_id, col_id in np.ndindex(original_weight.shape):
        if not is_watermark_row(args, layer_id, row_id, layer_name):
            continue
        if detect_mask is not None and not detect_mask[row_id][col_id]:
            continue
        if original_weight[row_id][col_id] == 0:
            continue
        use_weight, bit_position, coeff = get_projection_bit_mapping(
            args, layer_id, row_id, col_id, bit_count, layer_name
        )
        if not use_weight:
            continue
        coordinate = (row_id, col_id, coeff)
        detect_groups[bit_position].append(coordinate)
        if edit_mask is None or edit_mask[row_id][col_id]:
            edit_groups[bit_position].append(coordinate)
    margin = float(getattr(args, "projection_margin", 0.5))
    max_update = float(getattr(args, "projection_max_update", 0.0))
    detect_counts = [len(coords) for coords in detect_groups]
    edit_counts = [len(coords) for coords in edit_groups]
    print(f'projection detect group counts: {detect_counts}')
    print(f'projection editable group counts: {edit_counts}')
    changed = 0
    for bit_position, target_bit in enumerate(watermark_bits):
        detect_coords = detect_groups[bit_position]
        edit_coords = edit_groups[bit_position]
        n_edit = len(edit_coords)
        if len(detect_coords) == 0:
            print(f'bit {bit_position}: skipped, empty projection detect group')
            continue
        if n_edit == 0:
            print(f'bit {bit_position}: skipped, empty projection editable group')
            continue
        projection = float(
            sum(coeff * float(original_weight[row_id][col_id]) for row_id, col_id, coeff in detect_coords)
        )
        target_sign = 1.0 if target_bit == "1" else -1.0
        signed_margin = target_sign * projection
        if signed_margin >= margin:
            print(
                f'bit {bit_position}: keep bit={target_bit} '
                f'projection={projection:.8f} signed_margin={signed_margin:.8f}'
            )
            continue
        gap = margin - signed_margin
        step = gap / float(n_edit)
        if max_update > 0:
            step = min(step, max_update)
        update = np.array(target_sign * step, dtype=original_weight.dtype).item()
        for row_id, col_id, coeff in edit_coords:
            original_weight[row_id][col_id] = np.array(
                original_weight[row_id][col_id] + coeff * update,
                dtype=original_weight.dtype,
            )
            changed += 1
        new_projection = float(
            sum(coeff * float(original_weight[row_id][col_id]) for row_id, col_id, coeff in detect_coords)
        )
        print(
            f'bit {bit_position}: embed bit={target_bit} '
            f'before_projection={projection:.8f} after_projection={new_projection:.8f} '
            f'step={step:.8f} changed={n_edit}'
        )
    print(f'projection changed coordinate visits: {changed}')
    return original_weight
