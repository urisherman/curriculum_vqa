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
        train_dataset = datasets.NLVRDataset.load(nlvr_root, 'dev')

        params = {
            'd': 12  # embedding dimension
        }

        vqa_model = model.build_model(train_dataset.vocab, params)
        my_trainer = trainer.Trainer(train_dataset.vocab.pad_index)

        optimizer = torch.optim.Adam(vqa_model.parameters(), lr=1e-4)

        my_trainer.train(vqa_model, train_dataset, optimizer)