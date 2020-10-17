import os
import unittest

import numpy as np

import pathlib

import torch

from cvqa import datasets, trainers, fairseq_misc, utils, viz, models2
from cvqa.curriculum import VQAInstanceDistribution
from cvqa.datasets import Curriculum
from cvqa.vis_models import StructuredImageModel

project_root = pathlib.Path(__file__).parent.parent.absolute()
data_root = os.path.join(project_root, 'data-bin')
clevr_root = os.path.join(data_root, 'CLEVR_small_6')
curriculum_root = os.path.join(data_root, 'basic_curriculum')
tensorboard_root = os.path.join(project_root, 'tensorboard-logs/tests')

seed = 1

B = 2


class Models2Test(unittest.TestCase):

    def setUp(self) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)

        prompts = [
            ('shape', 'There is a [shape].', 'TRUE'),
            ('shape', 'There is a [color] item.', 'TRUE'),
            ('shape', 'There is no [shape].', 'FALSE'),
            ('shape', 'There is no [color] item', 'FALSE')
        ]
        vqa_dist = VQAInstanceDistribution(prompts)
        self.train_samples = Curriculum.process(vqa_dist.sample_dataset(10, 5))
        self.dev_samples = Curriculum.process(vqa_dist.sample_dataset(2, 4))

    def get_datasets(self):
        train_dataset = datasets.BaseDataset('xxx', self.train_samples, None)
        dev_dataset = datasets.BaseDataset('xxx', self.dev_samples, None, vocabs_from=train_dataset)
        train_dataset.use_viz_rep = True
        dev_dataset.use_viz_rep = True

        train_dataset.debug_mode = True
        dev_dataset.debug_mode = True

        return train_dataset, dev_dataset

    def get_clevr_datasets(self):
        train_dataset = datasets.CLEVR(clevr_root, 'train')
        dev_dataset = datasets.CLEVR(clevr_root, 'val', vocabs_from=train_dataset)
        train_dataset.use_viz_rep = True
        dev_dataset.use_viz_rep = True
        train_dataset.debug_mode = True
        dev_dataset.debug_mode = True
        return train_dataset, dev_dataset

    def test_clevr_train(self):
        train_dataset, dev_dataset = self.get_datasets()
        vocab = train_dataset.vocab
        ans_vocab = train_dataset.ans_vocab

        args = models2.default_args()

        img_model = StructuredImageModel(train_dataset.struct_viz_vocab, args['d_o'])
        model = models2.MostBasicModel(vocab, ans_vocab, img_model, args)

        my_trainer = trainers.VQATrainer(log_dir=tensorboard_root)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        my_trainer.train(model, train_dataset, dev_dataset, optimizer, num_epochs=2, batch_size=B)

    def test_training(self):

        train_dataset, dev_dataset = self.get_datasets()
        vocab = train_dataset.vocab
        ans_vocab = train_dataset.ans_vocab

        args = models2.default_args()

        img_model = StructuredImageModel(train_dataset, args['d_o'])
        model = models2.MostBasicModel(vocab, ans_vocab, img_model, args)

        my_trainer = trainers.VQATrainer(log_dir=tensorboard_root)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        my_trainer.train(model, train_dataset, dev_dataset, optimizer, num_epochs=2, batch_size=B)
