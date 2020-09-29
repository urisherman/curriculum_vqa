import os
import unittest

import numpy as np

import torch.nn.functional as F

import pathlib

import torch
from fairseq.data import Dictionary

from cvqa import datasets, models, trainers, fairseq_misc, utils, viz
from cvqa.models import BasicImgModel, StructuredImageModel

project_root = pathlib.Path(__file__).parent.parent.absolute()
data_root = os.path.join(project_root, 'data-bin')
nlvr_root = os.path.join(data_root, 'nlvr')
curriculum_root = os.path.join(data_root, 'basic_curriculum')
seed = 1

d_model = 16
B = 32


class VQAV11Test(unittest.TestCase):

    def setUp(self) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    def test_trainin_no_img(self):
        train_dataset = datasets.Curriculum(curriculum_root, 'train', limit=20)
        dev_dataset = datasets.Curriculum(curriculum_root, 'dev', vocab=train_dataset.vocab,  limit=5)

        my_trainer = trainers.VQATrainer()

        model = models.VQAModelV11.build(train_dataset.vocab, None, d_model=d_model)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        my_trainer.train(model, train_dataset, dev_dataset, optimizer, num_epochs=2, batch_size=B)

        y_true, y_pred = my_trainer.get_predictions(model, dev_dataset)

        # print(y_true)
        # print(y_pred)

    def test_training_with_structured_img(self):
        train_dataset = datasets.Curriculum(curriculum_root, 'train', limit=50)
        dev_dataset = train_dataset
        train_dataset.use_viz_rep = True

        my_trainer = trainers.VQATrainer()

        img_perceptor = StructuredImageModel(train_dataset.struct_viz_vocab, d_model, 2)
        model = models.VQAModelV11.build(train_dataset.vocab, img_perceptor, d_model=d_model)

        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        my_trainer.train(model, train_dataset, dev_dataset, optimizer, num_epochs=50, batch_size=B)

        y_true, y_pred = my_trainer.get_predictions(model, dev_dataset)
        # print(y_true)
        # print(y_pred)
