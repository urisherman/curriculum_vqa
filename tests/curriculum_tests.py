import os
import unittest
import numpy as np


import pathlib

from cvqa.curriculum import VQAInstanceDistribution

import pprint
pp = pprint.PrettyPrinter(indent=4)

project_root = pathlib.Path(__file__).parent.parent.absolute()
data_root = os.path.join(project_root, 'data-bin')
curriculum_root = os.path.join(data_root, 'basic_curriculum')
seed = 1


class CurriculumTest(unittest.TestCase):

    def test_vqa_dist(self):

        vqa_dist = VQAInstanceDistribution()
        vizrep = vqa_dist.sample_viz_rep()
        s = vqa_dist.sample_prompt(vizrep)

        pp.pprint(vizrep)
        pp.pprint(s)

    def test_vqa_dist_rep1(self):
        vqa_dist = VQAInstanceDistribution()

        viz_rep = {'color': 'blue', 'location': [0.65, 0.44], 'shape': 'triangle', 'size': 0.29}
        p = "This is a not a [color] item."
        ans = 'Wrong'
        s = vqa_dist.populate(p, ans, viz_rep)
        pp.pprint(s)

        # [{'answer': 'Right',
        #   'prompt': 'This is a not a blue object.',
        #   'realized_synonyms': {'image': 'image', 'item': 'object'}}]