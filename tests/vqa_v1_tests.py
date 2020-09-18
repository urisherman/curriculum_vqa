import os
import unittest
import numpy as np

import torchvision as tv

import pathlib

import torch

from cvqa import datasets, models, trainers
from cvqa.models import BasicImgModel

project_root = pathlib.Path(__file__).parent.parent.absolute()
data_root = os.path.join(project_root, 'data-bin')
nlvr_root = os.path.join(data_root, 'nlvr')
curriculum_root = os.path.join(data_root, 'basic_curriculum')
seed = 1

tensorboard_root = os.path.join(project_root, 'tensorboard-logs/tests')


class VQAV1Test(unittest.TestCase):

    def test_basic_img_model(self):
        np.random.seed(seed)
        torch.manual_seed(seed)

        d = 12

        B, C, H, W = 32, 3, 224, 224

        single_channel_model = BasicImgModel(d)
        img_embed = single_channel_model(torch.rand(B, C, H, W))
        self.assertEqual(list(img_embed.shape), [B, d], 'Wrong output dims of single channel img model')

        C_out = 16
        multi_channel_model = BasicImgModel(d, output_channels=C_out)
        img_embed = multi_channel_model(torch.rand(B, C, H, W))
        self.assertEqual(list(img_embed.shape), [B, C_out, d], 'Wrong output dims of multi channel img model')

    def test_training(self):
        train_dataset = datasets.BasicCurriculum(curriculum_root, 'train',
                                                 prompt_mode='natural', target_mode='natural', limit=50)
        dev_dataset = datasets.BasicCurriculum(curriculum_root, 'dev', vocab=train_dataset.vocab,
                                               prompt_mode='natural', target_mode='natural', limit=10)

        np.random.seed(seed)
        torch.manual_seed(seed)
        my_trainer = trainers.VQATrainer(log_dir=tensorboard_root)

        model = models.VQAModelV1.build(train_dataset.vocab, d=16, img_output_features=7)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        my_trainer.train(model, train_dataset, dev_dataset, optimizer, num_epochs=2, batch_size=5)
