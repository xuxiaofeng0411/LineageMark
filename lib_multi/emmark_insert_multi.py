import functools
import gc
import time
from collections import defaultdict

import numpy as np
import torch
import tqdm

from .data_multi import get_loaders
from .utils_multi import (
    find_layers,
    format_time,
    get_blocks,
    prepare_calibration_input,
    qweight2weight,
    string_to_binary,
    weight2qweight,
)


def _is_quantized_module(module):
    return hasattr(module, "qweight") and hasattr(module, "bits")


def _read_weight(module):
    if _is_quantized_module(module):
        return qweight2weight(module).data, True
    return module.weight.data, False


def _write_weight(module, modified_weight_tensor, is_quantized):
    if is_quantized:
        module.qweight.data = weight2qweight(modified_weight_tensor, module).to(module.qweight.device)
    else:
        module.weight.data = modified_weight_tensor.to(module.weight.device)


def insert_watermark(args, model, tokenizer, device=torch.device("cuda"), dataset_name="wikitext2"):
    watermark_bits = string_to_binary(args.watermark)
    watermark_length = len(watermark_bits)
    rng = np.random.default_rng(seed=args.seed)

    print("Loading calibdation data...")
    dataloader, _ = get_loaders(
        dataset_name,
        nsamples=args.nsamples,
        seed=args.seed,
        seqlen=model.seqlen,
        tokenizer=tokenizer,
    )
    with torch.no_grad():
        inps, outs, attention_mask, position_ids = prepare_calibration_input(model, dataloader, device, args)

    inps.to(device)
    outs.to(device)
    attention_mask.to(device)
    if position_ids is not None:
        position_ids.to(device)

    blocks = get_blocks(model)
    layer_num = sum(len(find_layers(block)) for block in blocks)
    every_layer_insert_bits_num = watermark_length // layer_num + 1
    modify_num = int(args.hidden_size * args.modify_rate)
    every_layer_modify_weight_num_per_bit = modify_num // every_layer_insert_bits_num
    every_layer_insert_candidate_bits_num = modify_num * args.candidate_rate
    print(f"every_layer_insert_bits_num: {every_layer_insert_bits_num}")

    insert_bit_position = 0
    change_weight_num = 0
    total_weight_num = 0
    total_indices = torch.empty((0, every_layer_insert_candidate_bits_num, 2), dtype=torch.long, device=device)
    layer_id = 0
    total_preprocess_time = 0.0
    total_insert_time = 0.0

    for block_index in tqdm.tqdm(range(len(blocks)), desc="Running EmMark insert..."):
        block = blocks[block_index].to(device)
        linear_layers = find_layers(block)

        preprocess_start = time.time()

        def cache_output_hook(module, input_value, output, name, feat_dict):
            del module, input_value
            if len(output.shape) == 3:
                output = output[0]
            feat_dict[name].append(output.clone().detach())

        output_feat = defaultdict(list)
        handles = []
        for name, module in linear_layers.items():
            handles.append(
                module.register_forward_hook(
                    functools.partial(cache_output_hook, name=name, feat_dict=output_feat)
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

        output_feat = {name: torch.mean(torch.stack(values, dim=0), dim=0) for name, values in output_feat.items()}
        total_preprocess_time += time.time() - preprocess_start

        for name, module in linear_layers.items():
            weight, is_quantized = _read_weight(module)
            print(f"[{layer_id}]: {name}: {module}")
            print("==> Get socres of weights...")
            preprocess_start = time.time()
            act = output_feat[name].to(weight.device)

            S_q = torch.abs(1 / weight)
            S_r = torch.abs(act.max() / (act - act.min()))
            S = S_q + S_r if S_r.shape[0] == S_q.shape[0] else S_q

            flattened_S = S.view(-1)
            values, flat_indices = torch.topk(flattened_S, every_layer_insert_candidate_bits_num, largest=False)
            del values
            indices = torch.unravel_index(flat_indices, S.shape)
            indices = torch.concat(indices, dim=0).view(2, -1).T.to(device)
            total_indices = torch.cat((total_indices, indices.unsqueeze(0)), dim=0)
            total_preprocess_time += time.time() - preprocess_start

            print("==> Insert watermark into weights...")
            insert_start = time.time()
            original_weight = weight.cpu()
            modified_weight = original_weight.detach().cpu().numpy()

            for bit_offset in range(every_layer_insert_bits_num):
                insert_bit = watermark_bits[(insert_bit_position + bit_offset) % watermark_length]
                for _ in range(every_layer_modify_weight_num_per_bit):
                    candidate_id = rng.integers(0, every_layer_insert_candidate_bits_num)
                    row_id, col_id = indices[candidate_id]
                    row_id = int(row_id.item())
                    col_id = int(col_id.item())
                    if insert_bit == "0":
                        modified_weight[row_id, col_id] += -1
                    else:
                        modified_weight[row_id, col_id] += 1
            insert_bit_position = (insert_bit_position + every_layer_insert_bits_num) % watermark_length

            weight_count = S.shape[0] * S.shape[1]
            modified_weight_tensor = torch.tensor(modified_weight, dtype=weight.dtype)
            _write_weight(module, modified_weight_tensor, is_quantized)
            difference = modified_weight_tensor - weight.cpu()
            non_zero_count = torch.count_nonzero(difference)
            print(f"==> Modify num: {non_zero_count}\tratio:{non_zero_count / weight_count}")

            total_insert_time += time.time() - insert_start
            change_weight_num += non_zero_count
            total_weight_num += weight_count
            layer_id += 1

        inps, outs = outs, inps
        del output_feat
        gc.collect()

    format_time(total_preprocess_time, "Preprocess")
    format_time(total_insert_time, "Insert")
    return change_weight_num, total_weight_num, total_indices
