import os
import unittest

import numpy as np

import torch.nn.functional as F

import pathlib

import torch
from fairseq.data import Dictionary

from cvqa import datasets, trainers, fairseq_misc, utils, viz, models2
from cvqa.curriculum import VQAInstanceDistribution
from cvqa.vis_models import StructuredImageModel

project_root = pathlib.Path(__file__).parent.parent.absolute()
data_root = os.path.join(project_root, 'data-bin')
nlvr_root = os.path.join(data_root, 'nlvr')
curriculum_root = os.path.join(data_root, 'basic_curriculum')
tensorboard_root = os.path.join(project_root, 'tensorboard-logs/tests')

seed = 1

B = 1


class Models2Test(unittest.TestCase):

    def setUp(self) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)

        prompts = [
            ('shape', 'There is a [shape].', 'TRUE'),
            ('shape', 'There is a [color] item.', 'TRUE'),
            ('shape', 'The item in the image is not a [shape]', 'FALSE'),
            ('shape', 'The thing you see is not [color]', 'FALSE')
        ]
        vqa_dist = VQAInstanceDistribution(prompts)
        self.train_samples = vqa_dist.sample_dataset(20, 4)
        self.dev_samples = vqa_dist.sample_dataset(5, 4)

    def get_datasets(self):
        train_dataset = datasets.BaseDataset('xxx', self.train_samples, None)
        vocab = train_dataset.vocab
        ans_vocab = train_dataset.ans_vocab
        dev_dataset = datasets.BaseDataset('xxx', self.dev_samples, None, vocab=vocab, ans_vocab=ans_vocab)
        train_dataset.use_viz_rep = True
        dev_dataset.use_viz_rep = True

        train_dataset.debug_mode = True
        dev_dataset.debug_mode = True

        return train_dataset, dev_dataset

    def test_training(self):

        train_dataset, dev_dataset = self.get_datasets()
        vocab = train_dataset.vocab
        ans_vocab = train_dataset.ans_vocab

        d = {
            'N_c': 14,

            'o': 16,
            'c': 40,
            'k': 15,

            'r': 41,
            'ak': 29,
            'av': 5,

            'w': 24,
            'a': 4
        }

        img_model = StructuredImageModel(train_dataset.struct_viz_vocab, d['o'])
        model = models2.MostBasicModel(vocab, ans_vocab, img_model, d)

        my_trainer = trainers.VQATrainer(log_dir=tensorboard_root)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        my_trainer.train(model, train_dataset, dev_dataset, optimizer, num_epochs=2, batch_size=B)
