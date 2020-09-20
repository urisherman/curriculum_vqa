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


class SanityTest(unittest.TestCase):

    def test_datasets(self):
        train_dataset = datasets.Curriculum(
            curriculum_root, 'train', prompt_mode='natural', target_mode='natural', download=False)
        # dev_dataset = datasets.BasicCurriculum(
        #     curriculum_root, 'dev', vocab=train_dataset.vocab, prompt_mode='natural', target_mode='natural')

        s = train_dataset[2]
        print(s['prompt'])
        print(train_dataset.samples[2]['encoded_prompt'])
        print(train_dataset.vocab.string(s['prompt']))

    # def test_vqa_training(self):
    #     np.random.seed(seed)
    #     torch.manual_seed(seed)
    #     train_dataset = datasets.NLVR(nlvr_root, 'train', limit=50)
    #     vocab = train_dataset.vocab
    #     dev_dataset = datasets.NLVR(nlvr_root, 'dev', vocab=vocab, limit=10)
    #
    #     params = {
    #         'd': 12  # embedding dimension
    #     }
    #
    #     vqa_model = models.VQAModelV0.build(vocab, params)
    #     my_trainer = trainers.VQATrainer(vocab.pad_index, log_dir=tensorboard_root)
    #
    #     optimizer = torch.optim.Adam(vqa_model.parameters(), lr=1e-4)
    #
    #     my_trainer.train(vqa_model, train_dataset, dev_dataset, optimizer, num_epochs=2, batch_size=10)

    def test_get_clf_predictions(self):
        train_dataset = datasets.Curriculum(curriculum_root, 'train', prompt_mode='concept', target_mode='class', limit=100)

        params = {
            'd': 12  # embedding dimension
        }

        viz_model = models.VQAPromptOpModel.build(2, train_dataset, c=5)
        my_trainer = trainers.ImageClassifierTrainer(log_dir=tensorboard_root)
        my_trainer.get_clf_predictions(viz_model, train_dataset)

    def __get_lesson1_datasets(self, prompt_mode='concept', target_mode='class'):
        train_dataset = datasets.Curriculum(curriculum_root, 'train', prompt_mode=prompt_mode, target_mode=target_mode, limit=50)
        dev_dataset = datasets.Curriculum(curriculum_root, 'dev', prompt_mode=prompt_mode, target_mode=target_mode, limit=10)
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