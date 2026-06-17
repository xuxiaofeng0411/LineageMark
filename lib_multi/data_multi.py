import random

import numpy as np
import torch
from datasets import load_dataset


WIKITEXT_SOURCE = ("Salesforce/wikitext", "wikitext-2-raw-v1")
C4_SOURCE = ("allenai/c4", "en")
C4_FILES = {
    "train": "en/c4-train.00000-of-01024.json.gz",
    "validation": "en/c4-validation.00000-of-00008.json.gz",
}
VALIDATION_BLOCKS = 256


def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)


class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids


def _labels_for_causal_lm(input_ids):
    labels = input_ids.clone()
    labels[:, :-1] = -100
    return labels


def _slice_token_window(encoded, start, seqlen):
    stop = start + seqlen
    return encoded.input_ids[:, start:stop]


def _draw_window(encoded, seqlen, rng):
    last_start = encoded.input_ids.shape[1] - seqlen - 1
    start = rng.randint(0, last_start)
    return _slice_token_window(encoded, start, seqlen)


def _make_training_pairs(encoded, nsamples, seqlen, rng):
    pairs = []
    for _ in range(nsamples):
        input_ids = _draw_window(encoded, seqlen, rng)
        pairs.append((input_ids, _labels_for_causal_lm(input_ids)))
    return pairs


def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    train_split = load_dataset(*WIKITEXT_SOURCE, split="train")
    test_split = load_dataset(*WIKITEXT_SOURCE, split="test")

    train_encoded = tokenizer(" ".join(train_split["text"]), return_tensors="pt")
    test_encoded = tokenizer("\n\n".join(test_split["text"]), return_tensors="pt")

    random.seed(seed)
    return _make_training_pairs(train_encoded, nsamples, seqlen, random), test_encoded


def _draw_c4_example(dataset, tokenizer, seqlen, rng):
    while True:
        row_id = rng.randint(0, len(dataset) - 1)
        encoded = tokenizer(dataset[row_id]["text"], return_tensors="pt")
        if encoded.input_ids.shape[1] > seqlen:
            break

    input_ids = _draw_window(encoded, seqlen, rng)
    return input_ids, _labels_for_causal_lm(input_ids)


def get_c4(nsamples, seed, seqlen, tokenizer):
    train_split = load_dataset(
        *C4_SOURCE,
        data_files={"train": C4_FILES["train"]},
        split="train",
    )
    validation_split = load_dataset(
        *C4_SOURCE,
        data_files={"validation": C4_FILES["validation"]},
        split="validation",
    )

    random.seed(seed)
    trainloader = [
        _draw_c4_example(train_split, tokenizer, seqlen, random)
        for _ in range(nsamples)
    ]

    validation_text = " ".join(validation_split[:1100]["text"])
    validation_ids = tokenizer(validation_text, return_tensors="pt").input_ids
    validation_ids = validation_ids[:, : VALIDATION_BLOCKS * seqlen]

    return trainloader, TokenizerWrapper(validation_ids)


def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    del name
    return get_wikitext2(
        nsamples=nsamples,
        seed=seed,
        seqlen=seqlen,
        tokenizer=tokenizer,
    )
