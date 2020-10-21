import os
import unittest

import numpy as np

import pathlib

import torch

from cvqa import datasets, trainers, fairseq_misc, utils, viz, models2
from cvqa.curriculum import VQAInstanceDistribution2
from cvqa.datasets import Curriculum
from cvqa.model_dev import answer_model
from cvqa.model_dev.lstms import Seq2SeqLSTM
from cvqa.vis_models import StructuredImageModel

project_root = pathlib.Path(__file__).parent.parent.absolute()
data_root = os.path.join(project_root, 'data-bin')
clevr_root = os.path.join(data_root, 'CLEVR_mini_6')
curriculum_root = os.path.join(data_root, 'basic_curriculum')
tensorboard_root = os.path.join(project_root, 'tensorboard-logs/tests')

seed = 1

B = 2


class Models2Test(unittest.TestCase):

    def setUp(self) -> None:
        np.random.seed(seed)
        torch.manual_seed(seed)

    def test_Seq2SeqLSTM_clevr(self):
        ds_train, ds_dev = datasets.CLEVR.load_train_dev(clevr_root)

        trainer = trainers.VQATrainer(pred_target='target_program_out', ignore_index=ds_train.vocab.pad_index)

        model_args = Seq2SeqLSTM.args(ds_train.vocab, ds_train.programs_vocab)
        model = Seq2SeqLSTM(model_args)
        optimizer = torch.optim.Adam(model.parameters(), lr=3e-3)

        trainer.train(model, ds_train, ds_dev, optimizer, num_epochs=2)

    def test_AnswerModel_curriculum(self):
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

    def test_MostBasicModel_curriculum(self):
        concept_dict = {
            'color': ['blue', 'brown', 'cyan', 'gray'],  # , 'green', 'purple', 'red', 'yellow'],
            'material': ['metal', 'rubber', 'plastic'],
            'shape': ['triangle', 'circle', 'square']
        }
        vqa_dist = VQAInstanceDistribution2(concept_dict)

        train_dataset, dev_dataset = datasets.Curriculum.from_samples(
            vqa_dist.sample_dataset(10, 5),
            vqa_dist.sample_dataset(2, 4)
        )

        vocab = train_dataset.vocab
        ans_vocab = train_dataset.ans_vocab

        args = models2.default_args()

        img_model = StructuredImageModel(args['d_o'])
        model = models2.MostBasicModel(vocab, ans_vocab, img_model, args)

        my_trainer = trainers.VQATrainer(log_dir=tensorboard_root)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        my_trainer.train(model, train_dataset, dev_dataset, optimizer, num_epochs=2, batch_size=B)
