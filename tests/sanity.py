import os
import unittest
import numpy as np

import pathlib

import torch

from cvqa import datasets, model, trainer

project_root = pathlib.Path(__file__).parent.parent.absolute()
data_root = os.path.join(project_root, 'data-bin')
nlvr_root = os.path.join(data_root, 'nlvr')
seed = 1


class SanityTest(unittest.TestCase):

    def test_0(self):
        np.random.seed(seed)
        torch.manual_seed(seed)
        train_dataset = datasets.NLVR(nlvr_root, 'train', limit=50)
        vocab = train_dataset.vocab
        dev_dataset = datasets.NLVR(nlvr_root, 'dev', vocab=vocab, limit=90)

        params = {
            'd': 12  # embedding dimension
        }

        vqa_model = model.build_model(vocab, params)
        my_trainer = trainer.Trainer(vocab.pad_index)

        optimizer = torch.optim.Adam(vqa_model.parameters(), lr=1e-4)

        my_trainer.train(vqa_model, train_dataset, dev_dataset, optimizer, num_epochs=2)

    def test_temp(self):
        x = []
        y = [10]

        def foo():
            x.append(1)
            x.append(y[0])

        for i in range(5):
            y[0] = 10*(i+1)
            foo()

        print(x)