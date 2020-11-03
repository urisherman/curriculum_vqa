import os
import unittest

import numpy as np

import pathlib

import torch
import torch.nn as nn

from cvqa import datasets, trainers, fairseq_misc, utils, viz, models2
from cvqa.curriculum import VQAInstanceDistribution2
from cvqa.datasets import Curriculum
from cvqa.model_dev import answer_model, f1_model, parent
from cvqa.model_dev.lstms import Seq2SeqLSTM
from cvqa.model_dev.parent import parse_dims_dict
from cvqa.model_dev.programs import ProgramSpec, Seq2ConstTreeModel, Seq2VecsLSTM
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
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer.train(model, ds_train, ds_dev, optimizer, num_epochs=2, batch_size=B)

    def test_F1Model_curriculum(self):
        concept_dict = {
            'color': ['blue', 'brown', 'cyan', 'gray'],
            'material': ['metal', 'rubber', 'plastic'],
            'shape': ['triangle', 'circle', 'square']
        }

        vqa_dist = VQAInstanceDistribution2(concept_dict=concept_dict, d_img=16, max_ref_concepts=1)
        ds_train, ds_dev = datasets.Curriculum.from_samples(
            vqa_dist.sample_dataset(images=100, prompts_per_image=3),
            vqa_dist.sample_dataset(images=20, prompts_per_image=3),
        )

        args = f1_model.default_args()
        model = f1_model.ParentModel(ds_train.vocab, ds_train.ans_vocab, args)

        trainer = trainers.VQATrainer(loss_fn=nn.NLLLoss(ignore_index=-1), pred_target='target_attention_mask', ignore_index=-1, progressbar='epochs')
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer.train(model, ds_train, ds_dev, optimizer, num_epochs=2, batch_size=B)

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

    def test_parent_curriculum(self):
        concept_dict = {
            'color': ['blue', 'brown', 'cyan', 'gray'],
            'material': ['metal', 'rubber', 'plastic'],
            'shape': ['triangle', 'circle', 'square']
        }

        vqa_dist = VQAInstanceDistribution2(concept_dict=concept_dict, d_img=24, max_ref_concepts=1)
        ds_train, ds_dev = datasets.Curriculum.from_samples(
            vqa_dist.sample_dataset(images=100, prompts_per_image=3),
            vqa_dist.sample_dataset(images=20, prompts_per_image=3),
        )

        prompt_vocab = ds_train.vocab
        ans_vocab = ds_train.ans_vocab

        args = parent.default_args()
        args['d_a'] = 4
        args['d_w'] = args['d_c'] = 32
        args['d_o'] = 24
        args['d_k'] = 4

        #### Build Model
        model_a = answer_model.AnswerModule(parse_dims_dict(args), ans_vocab)
        model_f1 = f1_model.F1ModuleSimple(args)

        #     program_spec = ProgramSpec({
        #         'A': model_a,
        #         'F': model_f1
        #     })
        #     seq2tree = Seq2ConstTreeModel(program_spec.vocab, 'A ( F )')

        program_spec = ProgramSpec({
            'M': parent.MultiModule(args['d_w'], [model_a, model_f1])
        })

        seq2tree = Seq2ConstTreeModel(program_spec.vocab, 'M ( M )')

        #     seeder_model = TransformerSeederModel(prompt_vocab, program_spec.vocab, args)

        seeder_args = Seq2VecsLSTM.args(prompt_vocab, program_spec.vocab)
        seeder_args['d_target'] = args['d_w']
        seeder_model = Seq2VecsLSTM(seeder_args)

        context_model = parent.ContextModel(args, ans_vocab)

        model = parent.MyModel(seq2tree, seeder_model, program_spec, context_model)
        # model = parent.MyModel.build(args, ds_train.vocab, ds_train.ans_vocab)
        ####

        trainer = trainers.VQATrainer(progressbar='epochs')
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer.train(model, ds_train, ds_dev, optimizer, num_epochs=2, batch_size=3)

        trainer.get_predictions(model, ds_train)

    def test_parent_clevr(self):
        from cvqa import clevr_utils

        ds_train, ds_dev = datasets.CLEVR.load_train_dev(clevr_root, programs_mapping=clevr_utils.programs_mapping, d_img=24, limit=50)
        clevr_utils.filter_samples(ds_train, ['f1'])
        clevr_utils.filter_samples(ds_dev, ['f1'])

        prompt_vocab = ds_train.vocab
        ans_vocab = ds_train.ans_vocab

        args = parent.default_args()
        args['d_a'] = 4
        args['d_w'] = args['d_c'] = 32
        args['d_o'] = 24
        args['d_k'] = 4

        #### Build Model
        model_a = answer_model.AnswerModule(parse_dims_dict(args), ans_vocab)
        model_f1 = f1_model.F1ModuleSimple(args)

        program_spec = ProgramSpec({
            'A': model_a,
            'F': model_f1
        })
        seq2tree = Seq2ConstTreeModel(program_spec.vocab, 'A ( F )')

        seeder_args = Seq2VecsLSTM.args(prompt_vocab, program_spec.vocab)
        seeder_args['d_target'] = args['d_w']
        seeder_model = Seq2VecsLSTM(seeder_args)

        context_model = parent.ContextModel(args, ans_vocab)

        model = parent.MyModel(seq2tree, seeder_model, program_spec, context_model)
        # model = parent.MyModel.build(args, ds_train.vocab, ds_train.ans_vocab)
        ####

        trainer = trainers.VQATrainer(progressbar='epochs')
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        trainer.train(model, ds_train, ds_dev, optimizer, num_epochs=2, batch_size=3)

        trainer.get_predictions(model, ds_train)