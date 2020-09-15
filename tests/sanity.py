import os
import unittest
import numpy as np

import torchvision as tv

import pathlib

import torch

from cvqa import datasets, model, trainer

project_root = pathlib.Path(__file__).parent.parent.absolute()
data_root = os.path.join(project_root, 'data-bin')
nlvr_root = os.path.join(data_root, 'nlvr')
curriculum_root = os.path.join(data_root, 'basic_curriculum')
seed = 1

tensorboard_root = os.path.join(project_root, 'tensorboard-logs/tests')


class SanityTest(unittest.TestCase):

    def test_vqa_training(self):
        np.random.seed(seed)
        torch.manual_seed(seed)
        train_dataset = datasets.NLVR(nlvr_root, 'train', limit=50)
        vocab = train_dataset.vocab
        dev_dataset = datasets.NLVR(nlvr_root, 'dev', vocab=vocab, limit=10)

        params = {
            'd': 12  # embedding dimension
        }

        vqa_model = model.build_model(vocab, params)
        my_trainer = trainer.VQATrainer(vocab.pad_index, log_dir=tensorboard_root)

        optimizer = torch.optim.Adam(vqa_model.parameters(), lr=1e-4)

        my_trainer.train(vqa_model, train_dataset, dev_dataset, optimizer, num_epochs=2, batch_size=10)

    def test_imgclf_training(self):
        np.random.seed(seed)
        torch.manual_seed(seed)
        train_dataset = datasets.BasicCurriculum(curriculum_root, 'train', limit=50)
        dev_dataset = datasets.BasicCurriculum(curriculum_root, 'dev', limit=10)

        viz_model = model.VQAConcept2ClassModel(len(train_dataset.concept_to_idx), len(train_dataset.cls_to_idx))

        optimizer = torch.optim.Adam(viz_model.parameters(), lr=1e-4)

        my_trainer = trainer.ImageClassifierTrainer(log_dir=tensorboard_root)
        my_trainer.train(viz_model, train_dataset, dev_dataset, optimizer, num_epochs=2, batch_size=16)