import os
import json
import torch

import torch.utils as torch_utils
import torch.nn.functional as F

from fairseq.data import Dictionary
from fairseq.tokenizer import tokenize_line


def encode_line(line, vocab, add_if_not_exist=True, consumer=None, append_eos=True, reverse_order=False):
    words = tokenize_line(line)
    if reverse_order:
        words = list(reversed(words))
    nwords = len(words)
    ids = torch.LongTensor(nwords + 1 if append_eos else nwords)

    for i, word in enumerate(words):
        if add_if_not_exist:
            idx = vocab.add_symbol(word)
        else:
            idx = vocab.index(word)
        if consumer is not None:
            consumer(word, idx)
        ids[i] = idx
    if append_eos:
        ids[nwords] = vocab.eos_index
    return ids


# class DatasetIterator:
#
#     def __init__(self, dataset, debug_mode=False, batch_size=32, shuffle=True):
#         self.debug_mode = debug_mode
#         self.dataset = dataset
#         self.loader = torch_utils.data.DataLoader(
#             dataset, batch_size=batch_size, shuffle=shuffle
#         )
#
#     def __iter__(self):
#         return self
#
#     def __next__(self):
#         return self.loader


class NLVRDataset(torch_utils.data.Dataset):

    @staticmethod
    def load(path, split):
        vocab = Dictionary()

        samples = []
        with open(os.path.join(path, split, f'{split}.json')) as data:
            for i, line in enumerate(data):
                sample = json.loads(line)
                samples.append(sample)
                sentence_ids = encode_line(sample['sentence'], vocab)
                sample['sentence_ids'] = sentence_ids

                label_ids = encode_line(sample['label'], vocab)
                sample['label_ids'] = label_ids

        vocab.finalize()
        return NLVRDataset(samples, vocab, os.path.join(path, split, 'images'))

    def __init__(self, samples, vocab, images_dir):
        self.samples = samples
        self.vocab = vocab
        self.images_dir = images_dir
        self.N = max(map(lambda s: len(s['sentence_ids']), samples))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        sample = self.samples[index]

        sentence = sample['sentence_ids']
        label = sample['label_ids']

        pad = self.N - len(sentence)
        sentence_tensor = F.pad(sentence, (0, pad), value=self.vocab.pad_index)
        labels_tensor = label

        return {
            'X': sentence_tensor,
            'target': labels_tensor,
            'question': sample['sentence'],
            'answer': sample['label']
        }

    def __repr__(self):
        S = len(self.samples)
        return f'Samples: {S} (N={self.N})\n' \
               f'Vocab Tokens:{len(self.vocab)}'



