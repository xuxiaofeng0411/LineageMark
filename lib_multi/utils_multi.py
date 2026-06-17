import hashlib
import traceback

import numpy as np
import torch
import torch.nn as nn
from auto_gptq.nn_modules.qlinear.qlinear_cuda_old import QuantLinear
from bitsandbytes.nn import Linear8bitLt
from transformers import AutoModelForCausalLM
from transformers.models.llama.modeling_llama import LlamaForCausalLM
from transformers.models.opt.modeling_opt import OPTForCausalLM
from transformers.models.phi3.modeling_phi3 import Phi3ForCausalLM


_SUPPORTED_GPTQ_BITS = (2, 4, 8)


def _is_llama_stack(model):
    return isinstance(model, (LlamaForCausalLM, Phi3ForCausalLM))


def _seeded_rng(*parts):
    return np.random.default_rng(seed=stable_int_hash(*parts))


def _np_scalar_from_binary(binary_string, dtype):
    byte_size = dtype.itemsize
    raw_value = int(binary_string, 2).to_bytes(byte_size, byteorder="little")
    if np.issubdtype(dtype, np.floating):
        return np.frombuffer(raw_value, dtype=dtype)[0]
    if np.issubdtype(dtype, np.integer):
        return np.array(int.from_bytes(raw_value, byteorder="little"), dtype=dtype)
    raise ValueError("Unsupported dtype")


def _coordinate_rng(args, tag, layer_id, row_id, col_id, layer_name):
    return _seeded_rng(args.password, tag, layer_id, layer_name, row_id, col_id)


def get_llm(model_name):
    load_kwargs = {"device_map": "auto"}
    if "8bit" not in model_name:
        load_kwargs["torch_dtype"] = torch.float16

    model = AutoModelForCausalLM.from_pretrained(model_name, **load_kwargs)
    model.seqlen = model.config.max_position_embeddings
    return model


def prepare_calibration_input(model, dataloader, device, args):
    layers = get_blocks(model)
    dtype = next(iter(model.parameters())).dtype
    inps = torch.zeros((128, model.seqlen, model.config.hidden_size), dtype=dtype, device=device)
    inps.requires_grad = False
    cache = {"i": 0, "attention_mask": None, "position_ids": None}

    class Catcher(nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, inp, **kwargs):
            cache_index = cache["i"]
            inps[cache_index] = inp
            cache["i"] = cache_index + 1
            cache["attention_mask"] = kwargs["attention_mask"]
            if not isinstance(model, OPTForCausalLM):
                cache["position_ids"] = kwargs["position_ids"]
            raise ValueError

    layers[0] = Catcher(layers[0])
    for batch in dataloader:
        try:
            model(batch[0].to(device))
        except ValueError:
            pass
    layers[0] = layers[0].module

    outs = torch.zeros_like(inps)
    return inps, outs, cache["attention_mask"], cache["position_ids"]


def get_blocks(model):
    if _is_llama_stack(model):
        return model.model.layers
    if isinstance(model, OPTForCausalLM):
        return model.model.decoder.layers
    if hasattr(model, "gpt_neox") and hasattr(model.gpt_neox, "layers"):
        return model.gpt_neox.layers
    raise NotImplementedError(type(model))


def find_layers(module, layers=[nn.Linear, Linear8bitLt, QuantLinear], name=''):
    if type(module) in layers:
        return {name: module}

    found = {}
    for child_name, child_module in module.named_children():
        qualified_name = child_name if name == '' else name + '.' + child_name
        child_layers = find_layers(child_module, layers=layers, name=qualified_name)
        if child_layers:
            found.update(child_layers)
    return found


def move_embed(model, device):
    if isinstance(model, LlamaForCausalLM):
        model.model.embed_tokens = model.model.embed_tokens.to(device)
        return
    if isinstance(model, OPTForCausalLM):
        decoder = model.model.decoder
        decoder.embed_tokens = decoder.embed_tokens.to(device)
        decoder.embed_positions = decoder.embed_positions.to(device)
        return
    raise NotImplementedError(type(model))


def get_position_bits(weight, num):
    return value_to_binary(weight)[:num]


def stable_int_hash(*parts):
    return int(hashlib.md5("|".join(str(part) for part in parts).encode()).hexdigest(), 16)


def is_watermark_row(args, layer_id, row_id, layer_name=""):
    gamma_row = max(1, args.hidden_size // 4)
    row_key = stable_int_hash(args.password, "wm-row-v2", layer_id, layer_name, row_id)
    return row_key % gamma_row == 0


def get_watermark_bit_mapping(args, layer_id, row_id, col_id, chunk_length, layer_name=""):
    rng = _coordinate_rng(args, "wm-bit-v2", layer_id, row_id, col_id, layer_name)
    if int(rng.integers(0, max(1, args.xi))) != 0:
        return False, None, None
    return True, int(rng.integers(0, args.xi)), int(rng.integers(0, chunk_length))


def get_mean_diff_group_mapping(args, layer_id, row_id, col_id, chunk_length, layer_name=""):
    rng = _coordinate_rng(args, "wm-mean-diff-v1", layer_id, row_id, col_id, layer_name)
    if int(rng.integers(0, max(1, args.xi))) != 0:
        return False, None, None
    return True, int(rng.integers(0, chunk_length)), int(rng.integers(0, 2))


def get_projection_bit_mapping(args, layer_id, row_id, col_id, chunk_length, layer_name=""):
    rng = _coordinate_rng(args, "wm-projection-v1", layer_id, row_id, col_id, layer_name)
    if int(rng.integers(0, max(1, args.xi))) != 0:
        return False, None, None
    bit_position = int(rng.integers(0, chunk_length))
    coeff = 1.0 if int(rng.integers(0, 2)) == 1 else -1.0
    return True, bit_position, coeff


def modify_bit_of_weight(weight, insert_bit, position):
    binary_representation = value_to_binary(weight)
    index = -(position + 1)
    replacement = str(insert_bit)
    if index == -1:
        rewritten = binary_representation[:index] + replacement
    else:
        rewritten = binary_representation[:index] + replacement + binary_representation[index + 1:]
    return binary_to_value(rewritten, weight.dtype)


def binary_to_value(binary_string, dtype):
    return _np_scalar_from_binary(binary_string, dtype)


def value_to_binary(weight):
    byte_size = weight.dtype.itemsize
    as_integer = int.from_bytes(weight.tobytes(), "little")
    return bin(as_integer)[2:].zfill(byte_size * 8)


def get_bit_from_weight(weight, position):
    return value_to_binary(weight)[-(position + 1)]


def string_to_binary(data):
    return ''.join(format(ord(char), '08b') for char in data)


def get_extract_acc(watermark_bits, extracted_watermark_bits):
    hit_num = sum(
        1
        for bit_index in range(len(watermark_bits))
        if watermark_bits[bit_index] == extracted_watermark_bits[bit_index]
    )
    return hit_num / len(watermark_bits)


def get_extract_chunk_acc(watermark_bits, extracted_watermark_bits):
    max_acc = 0
    max_acc_id = -1
    for chunk_id, offset in enumerate(range(0, (len(watermark_bits) // 8) * 8, 8)):
        hit_num = 0
        for bit_id in range(8):
            if watermark_bits[offset + bit_id] == extracted_watermark_bits[bit_id]:
                hit_num += 1
        chunk_acc = hit_num / 8
        if chunk_acc > max_acc:
            max_acc = chunk_acc
            max_acc_id = chunk_id
    return max_acc, max_acc_id


def qweight2weight(qlinear: QuantLinear):
    if qlinear.wf.device != qlinear.qzeros.device:
        qlinear.wf = qlinear.wf.to(qlinear.qzeros.device)

    if qlinear.bits in _SUPPORTED_GPTQ_BITS:
        shifts = qlinear.wf.unsqueeze(-1)
        expanded = torch.unsqueeze(qlinear.qweight, 1).expand(-1, 32 // qlinear.bits, -1)
        unpacked = torch.bitwise_right_shift(expanded, shifts)
        unpacked = unpacked.to(torch.int16 if qlinear.bits == 8 else torch.int8)
        unpacked = torch.bitwise_and(unpacked, (2 ** qlinear.bits) - 1)
        return unpacked.reshape(-1, unpacked.shape[2]).to(torch.uint8)


def weight2qweight(weight, qlinear: QuantLinear):
    intweight = weight.cpu().numpy().astype(np.uint32)
    rows_per_pack = 32 // qlinear.bits
    qweight = np.zeros((intweight.shape[0] // 32 * qlinear.bits, intweight.shape[1]), dtype=np.uint32)

    pack_start = 0
    packed_row = 0
    while packed_row < qweight.shape[0]:
        if qlinear.bits in _SUPPORTED_GPTQ_BITS:
            packed = np.zeros((intweight.shape[1],), dtype=np.uint32)
            for source_row in range(pack_start, pack_start + rows_per_pack):
                packed |= intweight[source_row] << (qlinear.bits * (source_row - pack_start))
            qweight[packed_row] = packed
            pack_start += rows_per_pack
            packed_row += 1

    return torch.from_numpy(qweight.astype(np.int32))


def format_time(elapsed_time, time_name):
    elapsed_time_ms = elapsed_time * 1000
    hours, hour_remainder = divmod(int(elapsed_time_ms), 3600000)
    minutes, minute_remainder = divmod(hour_remainder, 60000)
    seconds, milliseconds = divmod(minute_remainder, 1000)
    print(f"{time_name} use time: {int(hours)}h {int(minutes)}m {int(seconds)}s {int(milliseconds)}ms")
