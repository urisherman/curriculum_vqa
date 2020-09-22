import json
import os
import pathlib
import shutil

import numpy as np
import random
import re

import matplotlib.pyplot as plt

from cvqa.curriculum import plotter


class VQAInstanceDistribution(object):

    def __init__(self):
        self.shapes = ['circle', 'triangle']
        self.colors = ['blue', 'red', 'grey']
        self.groundings = {
            'shape': self.shapes,
            'color': self.colors
        }

        self.prompts = [
            ('shape', 'Which shape is this?', '[shape]'),
            ('shape', 'This item is a...?', '[shape]'),
            ('shape', 'The item in the image is a...?', '[shape]'),
            ('shape', 'The shape of the item in the image is a...?', '[shape]'),
            ('shape_yes_no', 'Is the item in the image a [shape]?', 'Yes'),
            ('shape_yes_no', 'This is a [shape].', 'True'),
            ('shape_yes_no', 'This is not a [shape].', 'Wrong'),
            ('color', 'What is the color of this item?', '[color]'),
            ('color', 'What color is this item?', '[color]'),
            ('color', 'The color of this item is...?', '[color]'),
            ('color', 'The color of the item in the image is...?', '[color]'),
            ('color_yes_no', 'Is the color of the item in the image [color]?', 'Yes'),
            ('color_yes_no', 'This is a [color] item.', 'True'),
            ('color_yes_no', 'This is a not a [color] item.', 'Wrong'),
        ]
        self.pos_neg = {
            'Yes': 'No',
            'Right': 'Wrong',
            'True': 'False'
        }
        self.neg_pos = {
            v: k for k, v in self.pos_neg.items()
        }

        self.synonyms = {
            'item': ['item', 'object', 'thing'],
            'image': ['image', 'picture']
        }

    def sample_viz_rep(self):
        loc = np.round(np.random.ranf(2) * 0.8 + 0.1, 2)
        max_size = min(min(loc), 1 - max(loc)) - .05
        size = np.round(np.random.ranf() * max_size + .05, 2)
        return {
            'shape': random.choice(self.shapes),
            'color': random.choice(self.colors),
            'location': loc.tolist(),
            'size': size
        }

    def __flip_answer(self, answer):
        if answer in self.pos_neg:
            return self.pos_neg[answer]
        elif answer in self.neg_pos:
            return self.neg_pos[answer]
        else:
            return None

    def populate(self, concept, prompt, answer, viz_rep):
        realized_synonyms = {
            k: random.choice(v) for k, v in self.synonyms.items()
        }

        for k, v in realized_synonyms.items():
            prompt = re.sub(k, v, prompt)

        flipped_ans = self.__flip_answer(answer)
        if flipped_ans is not None:
            flip_ans = random.choice([True, False])
            answer = flipped_ans if flip_ans else answer

        for k, v in self.groundings.items():
            if flipped_ans is not None:
                if flip_ans:
                    # eg [shape] --> circle (when shape is actually a triangle)
                    realized_grounding = random.choice([grounded for grounded in v if grounded != viz_rep[k]])
                else:
                    # eg [shape] --> triangle
                    realized_grounding = viz_rep[k]
                prompt = re.sub(f'\[{k}\]', realized_grounding, prompt)
            else:
                answer = re.sub(f'\[{k}\]', viz_rep[k], answer)

        return {
            'concept': concept,
            'prompt': prompt,
            'target': answer
        }

    def sample_prompt(self, viz_rep, n=1):
        ret = []
        prompt_templates = random.sample(self.prompts, n)
        for concept, prompt, answer in prompt_templates:
            ret.append(self.populate(concept, prompt, answer, viz_rep))
        return ret

    def sample_dataset(self, images=10, prompts_per_image=5):
        dataset = []
        for i in range(images):
            img = self.sample_viz_rep()
            rel_img_path = f'images/img_{i}.png'

            prompts = self.sample_prompt(img, n=prompts_per_image)
            for p in prompts:
                vqa_sample = {
                    'viz_rep': img,
                    'image_path': rel_img_path
                }
                vqa_sample.update(p)
                dataset.append(vqa_sample)

        random.shuffle(dataset)
        return dataset

    def generate_dataset(self, root, images=10, prompts_per_image=5):

        images_root = os.path.join(root, 'images')
        os.makedirs(images_root)

        dataset = []
        for i in range(images):
            img = self.sample_viz_rep()
            fig = plotter.draw(img)
            rel_img_path = f'images/img_{i}.png'
            plt.savefig(os.path.join(root, rel_img_path))
            plt.close(fig)

            prompts = self.sample_prompt(img, n=prompts_per_image)
            for p in prompts:
                vqa_sample = {
                    'viz_rep': img,
                    'image_path': rel_img_path
                }
                vqa_sample.update(p)
                dataset.append(vqa_sample)

        random.shuffle(dataset)
        with open(os.path.join(root, 'dataset.json'), 'w') as f:
            json.dump(dataset, f)

