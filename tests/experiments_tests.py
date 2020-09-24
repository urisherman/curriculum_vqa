import os
import unittest
import numpy as np


import pathlib

from cvqa.curriculum import VQAInstanceDistribution

import pprint

from cvqa.experiments import Experiments

pp = pprint.PrettyPrinter(indent=4)

# project_root = pathlib.Path(__file__).parent.parent.absolute()
# data_root = os.path.join(project_root, 'data-bin')
# curriculum_root = os.path.join(data_root, 'basic_curriculum')
seed = 1


class ExperimentsTest(unittest.TestCase):

    def test_experiments(self):
        exp = Experiments(
            lambda d, img_output_features, encoder_ffn_dim: None,
            lambda model: None,
            save_checkpoints=False
        )
        exp.set_log_enabled(False)

        exp.execute({
            'd': [16, 32, 64, 128],
            'img_output_features': np.arange(2, 7),
            'encoder_ffn_dim': [16, 32, 64, 128]
        }, limit=10)

    def test_considered(self):
        exp = Experiments(
            lambda d: None,
            lambda model, lr: None,
            save_checkpoints=False
        )
        exp.set_log_enabled(False)

        try:
            exp.execute({
                'd': [16, 32, 64, 128],
                'img_output_features': np.arange(2, 7),
                'lr': [16, 32, 64, 128]
            }, limit=10)
            raise AssertionError('Should have got exception from experimenter')
        except ValueError:
            pass
