import os
import numpy as np
import random
import torch
from datasets import load_dataset

def set_seed(seed):
    np.random.seed(seed)
    torch.random.manual_seed(seed)

class TokenizerWrapper:
    def __init__(self, input_ids):
        self.input_ids = input_ids

def get_wikitext2(nsamples, seed, seqlen, tokenizer):
    traindata = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='train')
    testdata = load_dataset('Salesforce/wikitext', 'wikitext-2-raw-v1', split='test')
    trainenc = tokenizer(" ".join(traindata['text']), return_tensors='pt')
    testenc = tokenizer("\n\n".join(testdata['text']), return_tensors='pt')
    random.seed(seed)
    trainloader = []
    for _ in range(nsamples):
        i = random.randint(0, trainenc.input_ids.shape[1] - seqlen - 1)
        j = i + seqlen
        inp = trainenc.input_ids[:, i:j]
        tar = inp.clone()
        tar[:, :-1] = -100
        trainloader.append((inp, tar))
    return trainloader, testenc

def get_loaders(name, nsamples=128, seed=0, seqlen=2048, tokenizer=None):
    return get_wikitext2(nsamples, seed, seqlen, tokenizer)
