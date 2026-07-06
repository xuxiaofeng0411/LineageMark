import hashlib
import time

import numpy as np
import torch
import tqdm

from .data_multi import get_loaders
from .layerwrapper_multi import WrappedGPT
from .utils_multi import (
    find_layers,
    format_time,
    get_bit_from_weight,
    get_blocks,
    get_ellmark_bit_mapping,
    is_ellmark_watermark_row,
    modify_bit_of_weight,
    prepare_calibration_input,
    qweight2weight,
    string_to_binary,
    weight2qweight,
)


def _read_weight(args, module):
    if "8bit" in args.model or hasattr(module, "qweight"):
        return qweight2weight(module).data.t(), True
    return module.weight.data, False


def _write_weight(module, modified_weight_tensor, is_quantized):
    if is_quantized:
        module.qweight.data = weight2qweight(modified_weight_tensor.t(), module).to(module.qweight.device)
    else:
        module.weight.data = modified_weight_tensor.to(module.weight.device)


def insert_watermark(args, model, tokenizer, device, dataset_name="wikitext2"):
    print("Loading calibdation data...")
    dataloader, _ = get_loaders(
        dataset_name,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )

    blocks = get_blocks(model)
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, args)

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

    for block_index in tqdm.tqdm(range(len(blocks)), desc="Running ELLMark insert..."):
        block = blocks[block_index]
        block.to(device)
        linear_layers = find_layers(block)

        preprocess_start = time.time()
        wrapped_layers = {
            name: WrappedGPT(module, layer_id, name, "8bit" in args.model or hasattr(module, "qweight"))
            for name, module in linear_layers.items()
        }

        handles = []
        for hook_name, wrapped in wrapped_layers.items():
            handles.append(
                linear_layers[hook_name].register_forward_hook(
                    lambda _module, inp, out, wrapped=wrapped: wrapped.add_batch(inp[0].data, out.data)
                )
            )

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

            print("==> Get socres of weights...")
            preprocess_start = time.time()
            weight, is_quantized = _read_weight(args, module)
            original_weight = weight.cpu()
            row_scale = torch.sqrt(wrapped_layers[name].scaler_row.reshape((1, -1))).to(original_weight.device)
            W_metric = torch.abs(original_weight) * row_scale
            W_mask = torch.zeros_like(W_metric, dtype=torch.bool)
            selected_columns = int(W_metric.shape[1] * args.select_ratio)
            indices = torch.sort(W_metric, dim=-1, stable=True)[1][:, :selected_columns]
            W_mask.scatter_(1, indices, True)
            total_preprocess_time += time.time() - preprocess_start

            print("==> Insert watermark into weights...")
            insert_start = time.time()
            modified_weight = change_weight(args, layer_id, insert_layer_num, original_weight, W_mask)
            modified_weight_tensor = torch.tensor(modified_weight, dtype=original_weight.dtype)
            _write_weight(module, modified_weight_tensor, is_quantized)

            difference = modified_weight_tensor - weight.to(modified_weight_tensor.device)
            non_zero_count = torch.count_nonzero(difference)
            weight_count = original_weight.shape[0] * original_weight.shape[1]
            print(f"==> Modify num: {non_zero_count}\tratio:{non_zero_count / weight_count}")

            total_insert_time += time.time() - insert_start
            change_weight_num += non_zero_count
            total_weight_num += weight_count
            layer_id += 1
            insert_layer_num += 1

        inps, outs = outs, inps

    format_time(total_preprocess_time, "Preprocess")
    format_time(total_insert_time, "Insert")
    torch.cuda.empty_cache()
    return change_weight_num, total_weight_num


def change_weight(args, layer_id, insert_layer_num, original_weight, W_mask):
    original_weight = original_weight.detach().cpu().numpy()
    watermark = args.watermark
    watermark_bits = string_to_binary(watermark[insert_layer_num % len(watermark)])
    print(f"=========[{layer_id}]========[{watermark_bits}]")
    bit_count = len(watermark_bits)

    zeros = [0] * bit_count
    ones = [0] * bit_count
    for row_id in range(original_weight.shape[0]):
        if not is_ellmark_watermark_row(args, layer_id, row_id):
            continue
        for col_id in range(original_weight.shape[1]):
            value = original_weight[row_id][col_id]
            use_weight, insert_position, bit_position = get_ellmark_bit_mapping(args, value, bit_count)
            if not use_weight:
                continue
            if get_bit_from_weight(value, insert_position) == "1":
                ones[bit_position] += 1
            else:
                zeros[bit_position] += 1

    print("current distribution: ")
    print(f"ones: {ones}")
    print(f"zeros: {zeros}")

    reverse_bits_list = [0] * bit_count
    for bit_position, target_bit in enumerate(watermark_bits):
        difference = abs(ones[bit_position] - zeros[bit_position]) // 2
        if target_bit == "0":
            if ones[bit_position] > zeros[bit_position]:
                reverse_bits_list[bit_position] += difference
                delta = (zeros[bit_position] + difference) // 10
                reverse_bits_list[bit_position] += max(delta, args.delta)
            else:
                more = zeros[bit_position] - ones[bit_position]
                delta = max(ones[bit_position] // 10, args.delta)
                if more < delta:
                    reverse_bits_list[bit_position] += delta - more
        else:
            if ones[bit_position] < zeros[bit_position]:
                reverse_bits_list[bit_position] += difference
                delta = (ones[bit_position] + difference) // 10
                reverse_bits_list[bit_position] += max(delta, args.delta)
            else:
                more = ones[bit_position] - zeros[bit_position]
                delta = max(zeros[bit_position] // 10, args.delta)
                if more < delta:
                    reverse_bits_list[bit_position] += delta - more

    print(reverse_bits_list)

    for row_id in range(original_weight.shape[0]):
        if not is_ellmark_watermark_row(args, layer_id, row_id):
            continue
        for col_id in range(original_weight.shape[1]):
            if not W_mask[row_id][col_id]:
                continue
            value = original_weight[row_id][col_id]
            use_weight, insert_position, bit_position = get_ellmark_bit_mapping(args, value, bit_count)
            if not use_weight:
                continue
            if sum(reverse_bits_list) == 0:
                return original_weight
            if reverse_bits_list[bit_position] == 0:
                continue
            target_bit = watermark_bits[bit_position]
            if get_bit_from_weight(value, insert_position) == target_bit:
                continue
            original_weight[row_id][col_id] = modify_bit_of_weight(value, target_bit, insert_position)
            reverse_bits_list[bit_position] -= 1

    return original_weight
