import torch
import torch.nn as nn
import torch.nn.functional as F

import random
import numpy as np


class Vocabulary(object):

    def __init__(self):
        self.unk_index = 0
        self.pad_index = 1
        self.bos_index = 2
        self.eos_index = 3
        self.tokens = {
            '<unk>': self.unk_index,
            '<pad>': self.pad_index,
            '<s>': self.bos_index,
            '</s>': self.eos_index,
        }
        self.idxs = None
        self.building = True

    def __len__(self):
        return len(self.tokens)

    def build(self):
        self.building = False
        self.idxs = {v: k for k, v in self.tokens.items()}
        self.N_tokens = len(self)

    def encode(self, tokens):
        ret = []
        for t in tokens:
            ret.append(self.encode_symbol(t))
        return ret

    def encode_symbol(self, token):
        ret = self.tokens.get(token, self.unk_index)
        if ret == self.unk_index and self.building:
            ret = len(self.tokens)
            self.tokens[token] = ret
        return ret

    def all_tokens(self):
        return list(self.tokens.keys())[3:]

    def decode(self, tokens):
        if type(tokens) == torch.Tensor:
            tokens = list(tokens.numpy().squeeze())
        if type(tokens) == np.ndarray:
            tokens = list(tokens.squeeze())

        if type(tokens) == list:
            return [self.idxs[t] for t in tokens]
        else:
            return self.idxs[tokens[0]]


def random_sentences(characters, N_samples=1000, sentence_len=5, N_words=100, word_len=4):
    vocab = Vocabulary()

    for i in range(N_words):
        w = ''.join(random.choices(characters, k=word_len))
        vocab.encode_symbol(w)

    vocab.build()
    words = vocab.all_tokens()
    samples = []
    samples_txt = []
    for i in range(N_samples):
        s = random.choices(words, k=sentence_len)
        samples_txt.append(' '.join(s))
        samples.append([vocab.bos_index] + vocab.encode(s) + [vocab.eos_index])

    return torch.tensor(samples), vocab, samples_txt