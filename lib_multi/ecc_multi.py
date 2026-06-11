import math
from .utils_multi import string_to_binary

def bits_to_bytes(bits):
    pad_len = (-len(bits)) % 8
    padded = bits + "0" * pad_len
    return bytearray(int(padded[i:i + 8], 2) for i in range(0, len(padded), 8)), pad_len

def bytes_to_bits(data):
    return "".join(f"{byte:08b}" for byte in data)

def bits_to_string(bits):
    usable_len = len(bits) // 8 * 8
    chars = []
    for i in range(0, usable_len, 8):
        chars.append(chr(int(bits[i:i + 8], 2)))
    return "".join(chars)

def split_chunks(bits, chunk_length):
    pad_len = (-len(bits)) % chunk_length
    padded = bits + "0" * pad_len
    return [padded[i:i + chunk_length] for i in range(0, len(padded), chunk_length)], pad_len

def get_bch(bch_m, bch_t):
    try:
        import bchlib
    except ImportError as exc:
        raise ImportError(
            "BCH ECC requires bchlib. Install it in the active environment with: "
            "pip install bchlib"
        ) from exc
    try:
        return bchlib.BCH(t=bch_t, m=bch_m)
    except TypeError:
        return bchlib.BCH(bch_m, bch_t)

def validate_bch_payload(bch, data_bit_len, bch_m, bch_t):
    max_data_bits = bch.n - bch.ecc_bits
    if data_bit_len > max_data_bits:
        raise ValueError(
            f"BCH(m={bch_m}, t={bch_t}) only supports {max_data_bits} data bits "
            f"after byte padding, but got {data_bit_len}. Increase --bch_m or "
            f"lower --bch_t."
        )

def bch_encode_bits(raw_bits, bch_m, bch_t):
    bch = get_bch(bch_m, bch_t)
    data, data_pad_len = bits_to_bytes(raw_bits)
    validate_bch_payload(bch, len(data) * 8, bch_m, bch_t)
    ecc = bytearray(bch.encode(data))
    coded_bits = bytes_to_bits(data + ecc)
    ecc_meta = {
        "ecc_type": "bch",
        "bch_m": bch_m,
        "bch_t": bch_t,
        "bch_n": bch.n,
        "bch_ecc_bits": bch.ecc_bits,
        "bch_ecc_bytes": bch.ecc_bytes,
        "raw_bit_len": len(raw_bits),
        "data_pad_len": data_pad_len,
        "data_byte_len": len(data),
        "ecc_byte_len": len(ecc),
        "coded_bit_len": len(coded_bits),
    }
    return coded_bits, ecc_meta

def bch_decode_bits(coded_bits, raw_bit_len, bch_m, bch_t):
    bch = get_bch(bch_m, bch_t)
    data_byte_len = math.ceil(raw_bit_len / 8)
    validate_bch_payload(bch, data_byte_len * 8, bch_m, bch_t)
    total_byte_len = data_byte_len + bch.ecc_bytes
    total_bit_len = total_byte_len * 8
    received_bits = coded_bits[:total_bit_len].ljust(total_bit_len, "0")
    received, _ = bits_to_bytes(received_bits)
    data = bytearray(received[:data_byte_len])
    ecc = bytearray(received[data_byte_len:total_byte_len])
    bitflips = bch.decode(data, ecc)
    decode_ok = bitflips >= 0
    if decode_ok:
        bch.correct(data, ecc)
    decoded_bits = bytes_to_bits(data)[:raw_bit_len]
    return decoded_bits, decode_ok, bitflips

def get_rs(rs_nsym):
    try:
        from reedsolo import RSCodec, ReedSolomonError
    except ImportError as exc:
        raise ImportError(
            "Reed-Solomon ECC requires reedsolo. Install it in the active "
            "environment with: pip install reedsolo"
        ) from exc
    return RSCodec(rs_nsym), ReedSolomonError

def rs_encode_bits(raw_bits, rs_nsym):
    if rs_nsym <= 0:
        raise ValueError("--rs_nsym must be positive")
    rs, _ = get_rs(rs_nsym)
    data, data_pad_len = bits_to_bytes(raw_bits)
    coded = bytearray(rs.encode(data))
    coded_bits = bytes_to_bits(coded)
    ecc_meta = {
        "ecc_type": "rs",
        "rs_nsym": rs_nsym,
        "raw_bit_len": len(raw_bits),
        "data_pad_len": data_pad_len,
        "data_byte_len": len(data),
        "ecc_byte_len": rs_nsym,
        "coded_bit_len": len(coded_bits),
        "max_correct_symbol_errors": rs_nsym // 2,
    }
    return coded_bits, ecc_meta

def rs_decode_bits(coded_bits, raw_bit_len, rs_nsym):
    if rs_nsym <= 0:
        raise ValueError("--rs_nsym must be positive")
    rs, ReedSolomonError = get_rs(rs_nsym)
    data_byte_len = math.ceil(raw_bit_len / 8)
    total_byte_len = data_byte_len + rs_nsym
    total_bit_len = total_byte_len * 8
    received_bits = coded_bits[:total_bit_len].ljust(total_bit_len, "0")
    received, _ = bits_to_bytes(received_bits)
    try:
        decoded, corrected, errata_pos = rs.decode(received)
        decode_ok = True
        errata_count = len(errata_pos)
        data = bytearray(decoded)
    except ReedSolomonError:
        decode_ok = False
        errata_count = -1
        data = bytearray(received[:data_byte_len])

    decoded_bits = bytes_to_bits(data)[:raw_bit_len]
    return decoded_bits, decode_ok, errata_count

def encode_watermark_bits(args):
    raw_bits = string_to_binary(args.watermark)
    watermark_bit_length = getattr(args, "watermark_bit_length", None)
    if watermark_bit_length is not None:
        if watermark_bit_length <= 0:
            raise ValueError("--watermark_bit_length must be positive")
        if watermark_bit_length > len(raw_bits):
            raise ValueError(
                f"--watermark_bit_length={watermark_bit_length} exceeds "
                f"available watermark bits {len(raw_bits)}"
            )
        raw_bits = raw_bits[:watermark_bit_length]
    ecc_type = getattr(args, "ecc_type", "none")

    if ecc_type == "none":
        coded_bits = raw_bits
        ecc_meta = {
            "ecc_type": "none",
            "raw_bit_len": len(raw_bits),
            "coded_bit_len": len(coded_bits),
        }
    elif ecc_type == "bch":
        coded_bits, ecc_meta = bch_encode_bits(
            raw_bits,
            bch_m=args.bch_m,
            bch_t=args.bch_t,
        )
    elif ecc_type == "rs":
        coded_bits, ecc_meta = rs_encode_bits(
            raw_bits,
            rs_nsym=args.rs_nsym,
        )
    else:
        raise ValueError(f"Unsupported ecc_type: {ecc_type}")
    chunks, chunk_pad_len = split_chunks(coded_bits, args.chunk_length)
    ecc_meta["chunk_length"] = args.chunk_length
    ecc_meta["chunk_pad_len"] = chunk_pad_len
    ecc_meta["num_chunks"] = len(chunks)
    return raw_bits, coded_bits, chunks, ecc_meta

def decode_watermark_bits(args, extracted_coded_bits):
    raw_bits = string_to_binary(args.watermark)
    watermark_bit_length = getattr(args, "watermark_bit_length", None)
    if watermark_bit_length is not None:
        raw_bits = raw_bits[:watermark_bit_length]
    ecc_type = getattr(args, "ecc_type", "none")
    if ecc_type == "none":
        decoded_bits = extracted_coded_bits[:len(raw_bits)]
        return decoded_bits, bits_to_string(decoded_bits), True, 0
    if ecc_type == "bch":
        decoded_bits, decode_ok, bitflips = bch_decode_bits(
            extracted_coded_bits,
            raw_bit_len=len(raw_bits),
            bch_m=args.bch_m,
            bch_t=args.bch_t,
        )
        return decoded_bits, bits_to_string(decoded_bits), decode_ok, bitflips
    if ecc_type == "rs":
        decoded_bits, decode_ok, errata_count = rs_decode_bits(
            extracted_coded_bits,
            raw_bit_len=len(raw_bits),
            rs_nsym=args.rs_nsym,
        )
        return decoded_bits, bits_to_string(decoded_bits), decode_ok, errata_count

    raise ValueError(f"Unsupported ecc_type: {ecc_type}")
