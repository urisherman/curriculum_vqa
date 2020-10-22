import json
import os

import numpy as np
import random
import re

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F

from cvqa.curriculum import plotter
from cvqa.curriculum.vizencoder import VizEncoder


class VQAInstanceDistribution2(object):

    def __init__(self, concept_dict, prompt_types_filter=None, d_img=16, N_max_objs=7, max_ref_concepts=2):

        self.concepts = concept_dict
        self.N_k = len(concept_dict)
        self.N_max_objs = N_max_objs
        self.prompt_types_filter = prompt_types_filter
        self.max_ref_concepts = max_ref_concepts
        self.synonyms = {
            'item': ['item', 'object', 'thing'],
            'image': ['image', 'picture']
        }

        self.vizenc = VizEncoder(concept_dict, numeric_fields=['size', 'location'], d_img=d_img-9)

    def sample_viz_rep(self):
        N_objs = random.randint(1, self.N_max_objs)
        scene_objects = []
        for i in range(N_objs):
            loc = np.round(np.random.ranf(2) * 0.8 + 0.1, 2)
            max_size = min(min(loc), 1 - max(loc)) - .1
            size = np.round(np.random.ranf() * max_size + .05, 2)

            obj = {
                'location': loc.tolist(),
                'size': size
            }
            for cc in self.concepts:
                cc_value = random.choice(self.concepts[cc])
                obj[cc] = cc_value
            scene_objects.append(obj)
        viz = {
            'objects': scene_objects
        }
        encoded_viz = self.vizenc.encode(viz)
        encoded_viz = F.pad(encoded_viz, [0, 0, 0, self.N_max_objs - N_objs], value=0)
        viz['encoded'] = encoded_viz
        objects_mask = torch.zeros(self.N_max_objs, dtype=torch.bool)
        objects_mask[:N_objs] = 1
        viz['objects_mask'] = objects_mask
        return viz

    def attention_questions(self, predicate, att_mask):
        P_str = ' '.join(predicate.values())

        def ref_str(c, P_str, obj_id=None):
            obj_ref = f'{P_str} {obj_id}' if obj_id is not None else f'{P_str}'
            if c == 1:
                return f'is {c} {obj_ref}'
            else:
                return f'are {c} {obj_ref}s'

        questions = []

        c = int(att_mask.sum())
        c_plus = c + random.randint(1, 3)
        c_minus = max(1, c - random.randint(1, 3))

        answer_true = 'True'
        answer_false = 'False'
        if c == 0:
            answer_true = 'False'
            answer_false = 'True'

        if 'shape' in predicate:
            questions.append(('exists', f'There is a {P_str}', answer_true))
            questions.append(('exists', f'There are no {P_str}s', answer_false))
            questions.append(('exists_count', f'There {ref_str(c, P_str)}', answer_true))

            questions.append(('count', f'How many {P_str}s are there ?', f'{c}'))
            questions.append(('exists_count', f'Are there {c} {P_str}s ?', 'True'))
            questions.append(('exists_count', f'Are there {c_plus} {P_str}s ?', 'False'))
            # if c > 0:
            #     questions.append((f'Are there {c_minus} {P_str}s ?', 'False'))
            # questions.append((f'Are there less than {c_plus} {P_str}s ?', 'True'))
            # if c > 2:
            #     questions.append((f'Are there less than {c_minus} {P_str}s ?', 'False'))

        else:
            questions.append(('exists', f'There is a {P_str} item', answer_true))
            questions.append(('exists', f'There are no {P_str} items', answer_false))
            questions.append(('exists_count', f'There {ref_str(c, P_str, "item")}', answer_true))

            questions.append(('count', f'How many {P_str} items are there ?', f'{c}'))
            questions.append(('exists_count', f'Are there {c} {P_str} items ?', 'True'))
            questions.append(('exists_count', f'Are there {c_plus} {P_str} items ?', 'False'))
            # if c > 0:
            #     questions.append((f'Are there {c_minus} {P_str} items ?', 'False'))
            # questions.append((f'Are there less than {c_plus} {P_str} items ?', 'True'))
            # if c > 2:
            #     questions.append((f'Are there less than {c_minus} {P_str} items ?', 'False'))

        return list(zip(questions, [att_mask]*len(questions)))

    def sample_positive_attention(self, viz):
        num_concepts = random.randint(1, self.max_ref_concepts)
        P_concepts = np.random.choice(list(self.concepts.keys()), size=num_concepts, replace=False)

        objects = viz['objects']

        P_obj = random.choice(objects)
        P_pos = {k: P_obj[k] for k in self.concepts if k in P_concepts}

        att_mask = torch.ones(self.N_max_objs)
        att_mask[len(objects):] = 0
        for i, o in enumerate(objects):
            for k in P_pos:
                if o[k] != P_pos[k]:
                    att_mask[i] = 0

        return P_pos, att_mask

    def matches_predicate(self, o, P):
        for k in P:
            if o[k] != P[k]:
                return False
        return True

    def sample_negative_attention(self, viz):
        num_concepts = random.randint(1, self.max_ref_concepts)
        for i in range(20):
            P_concepts = np.random.choice(list(self.concepts.keys()), size=num_concepts, replace=False)

            P_neg = {}
            for ck in self.concepts:
                if ck in P_concepts:
                    cv = random.choice(self.concepts[ck])
                    P_neg[ck] = cv

            objects = viz['objects']
            matching_objects = [o for o in objects if self.matches_predicate(o, P_neg)]
            if len(matching_objects) == 0:
                return P_neg
        return None

    def attr_questions(self, objects):
        concept_dict = self.concepts

        questions = []
        for question_k in concept_dict:
            refs_dict = {}
            num_refs = random.randint(1, self.N_k)
            o_concept_refs = np.random.choice(list(concept_dict.keys()), size=num_refs, replace=False)
            o_concept_refs = [k for k in concept_dict.keys() if k in o_concept_refs and k != question_k]
            if len(o_concept_refs) > 0:
                for i, o in enumerate(objects):
                    P_str = ' '.join([o[k] for k in o_concept_refs])
                    if P_str not in refs_dict:
                        refs_dict[P_str] = []
                    refs_dict[P_str].append(i)

                item_str = 'item ' if 'shape' not in o_concept_refs else ''
                for r in refs_dict:
                    if len(refs_dict[r]) == 1:
                        i = refs_dict[r][0]
                        o = objects[i]
                        att_mask = torch.zeros(self.N_max_objs)
                        att_mask[i] = 1
                        questions.append((('retrieve_attribute', f'What {question_k} is the {r} {item_str}?', o[question_k]), att_mask))
                        # questions.append((('is_attribute', f'Is the {r} {item_str} of {}?', o[question_k]), att_mask))

        return questions

    def apply_prompt_type_filter(self, questions):
        if self.prompt_types_filter is None:
            return questions
        else:
            return [q for q in questions if q[0][0] in self.prompt_types_filter]

    def sample_prompt(self, viz_rep, n=1):
        P_pos, att_mask = self.sample_positive_attention(viz_rep)
        P_neg = self.sample_negative_attention(viz_rep)

        def choose_randomly(q_list):
            n_1 = min(n // 3, len(q_list))
            if n_1 > 0:
                return random.choices(q_list, k=n_1)
            else:
                return []

        q_pos_att = self.attention_questions(P_pos, att_mask)
        q_pos_att = self.apply_prompt_type_filter(q_pos_att)
        sampled_questions = choose_randomly(q_pos_att)

        q_attrs = self.attr_questions(viz_rep['objects'])
        q_attrs = self.apply_prompt_type_filter(q_attrs)
        sampled_questions += choose_randomly(q_attrs)

        if P_neg is not None:
            q_neg_att = self.attention_questions(P_neg, torch.zeros(self.N_max_objs))
            q_neg_att = self.apply_prompt_type_filter(q_neg_att)
            sampled_questions += choose_randomly(q_neg_att)

        ret = []
        for q in sampled_questions:
            att_mask = q[1]
            att_mask[~viz_rep['objects_mask']] = -1
            ret.append({
                'prompt_type': q[0][0],
                'prompt': q[0][1],
                'target': q[0][2],
                'attention_mask': att_mask.long()
            })
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
                self.add_debug_info(vqa_sample)
                dataset.append(vqa_sample)

        # random.shuffle(dataset)
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

    def add_debug_info(self, x):
        a = np.array([o['color'] + ' ' + o['material'] + ' ' + o['shape'] for o in x['viz_rep']['objects']])
        scene_text_arr = np.stack([x['attention_mask'][:len(a)].numpy().astype(int), a]).T
        prompt_answer = x['prompt'] + ' --> ' + x['target']
        x['debug_info'] = {
            'TYPE': x['prompt_type'],
            'PROMPT': prompt_answer,
            'SCENE ': scene_text_arr
        }


if __name__ == '__main__':
    concept_dict = {
        'color': ['blue', 'brown', 'cyan', 'gray'],  # , 'green', 'purple', 'red', 'yellow'],
        'material': ['metal', 'rubber', 'plastic'],
        'shape': ['triangle', 'circle', 'square']
    }

    vqa_dist = VQAInstanceDistribution2(concept_dict=concept_dict)
    ds = vqa_dist.sample_dataset(images=10, prompts_per_image=3)
    print(VQAInstanceDistribution2.to_debug_rep(ds))