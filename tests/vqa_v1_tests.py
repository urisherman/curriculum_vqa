import os
import unittest

import numpy as np

import torch.nn.functional as F

import pathlib

import torch
from fairseq.data import Dictionary

from cvqa import datasets, models, trainers, fairseq_misc, utils, viz
from cvqa.models import BasicImgModel

project_root = pathlib.Path(__file__).parent.parent.absolute()
data_root = os.path.join(project_root, 'data-bin')
nlvr_root = os.path.join(data_root, 'nlvr')
curriculum_root = os.path.join(data_root, 'basic_curriculum')
seed = 1

tensorboard_root = os.path.join(project_root, 'tensorboard-logs/tests')


class VQAV1Test(unittest.TestCase):

    def setUp(self) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    def test_basic_img_model(self):
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
        train_dataset = datasets.Curriculum(curriculum_root, 'train', limit=20)
        dev_dataset = datasets.Curriculum(curriculum_root, 'dev', vocab=train_dataset.vocab,  limit=5)

        my_trainer = trainers.VQATrainer(log_dir=tensorboard_root)

        model = models.VQAModelV1.build(train_dataset.vocab, d=16, img_output_features=7)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        my_trainer.train(model, train_dataset, dev_dataset, optimizer, num_epochs=2, batch_size=6)

        y_true, y_pred = my_trainer.get_predictions(model, dev_dataset)

        # print(y_true)
        # print(y_pred)

    def test_training_with_structured_img(self):
        train_dataset = datasets.Curriculum(curriculum_root, 'train', limit=50)
        dev_dataset = train_dataset
        train_dataset.use_viz_rep = True

        my_trainer = trainers.VQATrainer(log_dir=tensorboard_root)
        model = models.VQAModelV1.struct_img_build(train_dataset, d=16, img_output_features=3)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        my_trainer.train(model, train_dataset, dev_dataset, optimizer, num_epochs=50, batch_size=32)

        y_true, y_pred = my_trainer.get_predictions(model, dev_dataset)
        # print(y_true)
        # print(y_pred)
