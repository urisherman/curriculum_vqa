import os
import unittest
import numpy as np

import pathlib

import torch

from cvqa import datasets, models, trainers

project_root = pathlib.Path(__file__).parent.parent.absolute()
data_root = os.path.join(project_root, 'data-bin')
nlvr_root = os.path.join(data_root, 'nlvr')
curriculum_root = os.path.join(data_root, 'basic_curriculum')
seed = 1

tensorboard_root = os.path.join(project_root, 'tensorboard-logs/tests')


class PromptOpTest(unittest.TestCase):

    def test_get_clf_predictions(self):
        train_dataset = datasets.Curriculum(curriculum_root, 'train', prompt_mode='concept', target_mode='class', limit=100)

        params = {
            'd': 12  # embedding dimension
        }

        viz_model = models.VQAPromptOpModel.build(2, train_dataset, c=5)
        my_trainer = trainers.ImageClassifierTrainer(log_dir=tensorboard_root)
        my_trainer.get_clf_predictions(viz_model, train_dataset)

    def __get_lesson1_datasets(self, prompt_mode='concept', target_mode='class'):
        train_dataset = datasets.Curriculum(curriculum_root, 'train', prompt_mode=prompt_mode, target_mode=target_mode, limit=20)
        dev_dataset = datasets.Curriculum(curriculum_root, 'dev', prompt_mode=prompt_mode, target_mode=target_mode, limit=5)
        return train_dataset, dev_dataset

    def test_promptop_training(self):
        np.random.seed(seed)
        torch.manual_seed(seed)
        my_trainer = trainers.ImageClassifierTrainer(log_dir=tensorboard_root)

        train_dataset, dev_dataset = self.__get_lesson1_datasets('concept', 'class')
        viz_model = models.VQAPromptOpModel.build(2, train_dataset, c=5)
        optimizer = torch.optim.Adam(viz_model.parameters(), lr=1.5e-3)
        my_trainer.train(viz_model, train_dataset, dev_dataset, optimizer, num_epochs=2, batch_size=16)

        train_dataset, dev_dataset = self.__get_lesson1_datasets('natural', 'class')
        viz_model = models.VQAPromptOpModel.build(2, train_dataset, c=5)
        optimizer = torch.optim.Adam(viz_model.parameters(), lr=1.5e-3)
        my_trainer.train(viz_model, train_dataset, dev_dataset, optimizer, num_epochs=2, batch_size=16)