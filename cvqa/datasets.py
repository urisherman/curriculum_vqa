import os
import json
import urllib
import zipfile
from pathlib import Path
import shutil

import gdown
import torch

import torch.utils as torch_utils
import torch.nn.functional as F

from fairseq.data import Dictionary
from fairseq.tokenizer import tokenize_line

import torchvision as tv


DATASETS = {
    'NLVR': {
        'url': 'https://drive.google.com/uc?id=1cVUPwPYIwvHY_TQxRQ9sSvpDZdxzTRXz'
    }
}


def download_if_needed(dataset_name, root, a_file):
    if not os.path.exists(a_file):
        shutil.rmtree(root, ignore_errors=True)
        os.makedirs(root)
        # print(f"Downloading {dataset_name} dataset...")
        dataset_zip = os.path.join(root, f'{dataset_name}.zip')
        gdown.download(DATASETS[dataset_name]['url'], dataset_zip, quiet=False)
        with zipfile.ZipFile(dataset_zip, 'r') as zip_ref:
            zip_ref.extractall(root)
        print(f'Dataset was downloaded successfully and extracted to {root}')


class NLVR(torch_utils.data.Dataset):

    @staticmethod
    def load(ds_file, vocab=None, limit=None):
        is_build_vocab = vocab is None
        if is_build_vocab:
            vocab = Dictionary()

        samples = []
        with open(ds_file) as data:
            for i, line in enumerate(data):
                sample = json.loads(line)
                samples.append(sample)
                sentence_ids = encode_line(sample['sentence'], vocab, add_if_not_exist=is_build_vocab)
                sample['sentence_ids'] = sentence_ids

                label_ids = encode_line(sample['label'], vocab)
                sample['label_ids'] = label_ids
                if limit is not None and i > limit:
                    break

        vocab.finalize()
        return samples, vocab

    def __init__(self, root, split='train', vocab=None, img_transform=None, limit=None, download=False):
        ds_file = os.path.join(root, split, f'{split}.json')
        if download:
            download_if_needed('NLVR', root, ds_file)
        samples, vocab = NLVR.load(ds_file, vocab=vocab, limit=limit)
        self.samples = samples
        self.vocab = vocab
        self.images_dir = os.path.join(root, split, 'images')
        self.split = split
        self.N = max(map(lambda s: len(s['sentence_ids']), samples))

        if img_transform is None:
            img_transform = tv.transforms.Compose([
                tv.transforms.Pad((0, 150), fill=300, padding_mode='constant'),
                tv.transforms.Resize(224),
                tv.transforms.ToTensor(),
                tv.transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
        self.img_transform = img_transform

    def __len__(self):
        return len(self.samples)

    def img_path(self, sample):
        img_id = sample['identifier']
        img_file = f'{self.split}-{img_id}-0.png'
        return os.path.join(self.images_dir, sample['directory'], img_file)

    def __getitem__(self, index):
        sample = self.samples[index]

        sentence = sample['sentence_ids']
        label = sample['label_ids']

        pad = self.N - len(sentence)
        sentence_tensor = F.pad(sentence, (0, pad), value=self.vocab.pad_index)
        labels_tensor = label

        sample_img = tv.datasets.folder.default_loader(self.img_path(sample))
        if self.img_transform is not None:
            sample_img = self.img_transform(sample_img)

        return {
            'X': sentence_tensor,
            'target': labels_tensor,
            'question': sample['sentence'],
            'answer': sample['label'],
            'img': sample_img
        }

    def __repr__(self):
        S = len(self.samples)
        return f'Samples: {S} (N={self.N})\n' \
               f'Vocab Tokens:{len(self.vocab)}'




# TODO: Sort this out - what is the proper way to encode and build the dictionary? Note - fairseq transformer expects int64 type token ids.
def encode_line(line, vocab, add_if_not_exist=True, consumer=None, append_eos=True, reverse_order=False):
    """
    Copied from fairseq.data.Dictionary and changed ids tensor type to Long (==int64)
    :param line:
    :param vocab:
    :param add_if_not_exist:
    :param consumer:
    :param append_eos:
    :param reverse_order:
    :return:
    """
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
