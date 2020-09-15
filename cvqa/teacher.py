import json
import os
import pathlib
import shutil
import random

import numpy as np
import matplotlib.pyplot as plt

import re

x = {
    'object': 'circle',
    'color': 'blue',
    'location': (.5, .5),
    'size': .2
}


# x = ['Is this a triangle?', 'Yes.',
#      'What color is this object?', 'Blue',
#      'Is this item black?', 'No.',
#      'This item is a ...', 'triangle!',
#      'This is not a square', 'True.',
#      'This is not a triangle', 'Wrong'
#      ]

QUESTIONS = [
    ('shape', 'Which shape is this?', '[shape]'),
    ('color', 'What is the color of this item?', '[color]')
]


def write_dataset(viz_reps, root):
    images_root = os.path.join(root, 'images')
    os.makedirs(images_root)

    dataset = []
    for i, viz in enumerate(viz_reps):
        fig = draw(viz)
        plt.savefig(os.path.join(root, f'images/img_{i}.png'))
        plt.close(fig)
        dataset += make_training_samples(viz, f'images/img_{i}.png')
    with open(os.path.join(root, 'dataset.json'), 'w') as f:
        json.dump(dataset, f)


def make_training_samples(viz_rep, image_path):

    def populate(qa_pair, viz_rep):
        for k in ['shape', 'color']:
            v = viz_rep[k]
            qa_pair[0] = re.sub(f'\[{k}\]', v, qa_pair[0])
            qa_pair[1] = re.sub(f'\[{k}\]', v, qa_pair[1])
        return qa_pair

    samples = []

    for q in QUESTIONS:
        qa_pair = [q[1], q[2]]
        populated_qa = populate(qa_pair, viz_rep)
        samples.append({
            'viz_rep': viz_rep,
            'image_path': image_path,
            'concept': q[0],
            'question': populated_qa[0],
            'answer': populated_qa[1]
        })

    return samples


def draw(sample):
    fig, ax = plt.subplots(figsize=(5, 5))
    {
        'circle': draw_circle,
        'triangle': draw_triangle
    }[sample['shape']](sample, ax)

    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis('off')
    return fig


def draw_circle(sample, ax):
    circ = plt.Circle(sample['location'], sample['size'], color=sample['color'])
    ax.add_artist(circ)


def draw_triangle(sample, ax):
    r = sample['size']
    center = sample['location']
    edge_length = r * 4 / np.sqrt(3)

    x = np.zeros(3)
    y = np.zeros(3)

    x[0] = center[0] - .5 * edge_length
    y[0] = center[1] - r

    x[1] = center[0] + .5 * edge_length
    y[1] = y[0]

    x[2] = center[0]
    y[2] = center[1] + r

    ax.scatter(x, y, s=0, color=sample['color'])
    t1 = plt.Polygon(np.hstack([x[:, None], y[:, None]]), color=sample['color'])
    ax.add_patch(t1)


def generate(split, num_images):
    project_root = pathlib.Path(__file__).parent.parent.absolute()
    dataset_root = os.path.join(project_root, f'data-bin/basic_curriculum/{split}')
    shutil.rmtree(dataset_root, ignore_errors=True)

    viz_list = []
    for i in range(num_images):
        loc = np.round(np.random.ranf(2) * 0.8 + 0.1, 2)
        max_size = min(min(loc), 1 - max(loc)) - .05
        size = np.round(np.random.ranf() * max_size + .05, 2)
        viz_list.append({
            'shape': random.sample(['circle', 'triangle'], 1)[0],
            'color': random.sample(['blue', 'red', 'grey'], 1)[0],
            'location': loc.tolist(),
            'size': size
        })

    write_dataset(viz_list, dataset_root)


if __name__ == '__main__':
    generate('train', 1000)
    generate('dev', 100)
