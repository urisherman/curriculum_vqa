import os
import unittest

import numpy as np

import pathlib

import torch

from cvqa import datasets, trainers, fairseq_misc, utils, viz, models2
from cvqa.curriculum import VQAInstanceDistribution, VQAInstanceDistribution2
from cvqa.datasets import Curriculum
from cvqa.model_dev import answer_model
from cvqa.vis_models import StructuredImageModel

project_root = pathlib.Path(__file__).parent.parent.absolute()
data_root = os.path.join(project_root, 'data-bin')
clevr_root = os.path.join(data_root, 'CLEVR_small_6')
curriculum_root = os.path.join(data_root, 'basic_curriculum')
tensorboard_root = os.path.join(project_root, 'tensorboard-logs/tests')

seed = 1

B = 2


class TempTest(unittest.TestCase):

    def setUp(self) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    def test_foo(self):
        concept_dict = {
            'color': ['blue', 'brown', 'cyan', 'gray'],  # , 'green', 'purple', 'red', 'yellow'],
            'material': ['metal', 'rubber', 'plastic'],
            'shape': ['triangle', 'circle', 'square']
        }

        vqa_dist = VQAInstanceDistribution2(concept_dict=concept_dict, d_img=16)
        ds_train, ds_dev = datasets.Curriculum.from_samples(
            vqa_dist.sample_dataset(images=100, prompts_per_image=3),
            vqa_dist.sample_dataset(images=20, prompts_per_image=3),
        )

        args = answer_model.default_args()
        model = answer_model.ParentModel(ds_train.vocab, ds_train.ans_vocab, args)

        trainer = trainers.VQATrainer(progressbar='epochs')
        print(trainer.evaluate(model, torch.utils.data.DataLoader(ds_train, batch_size=B)))

