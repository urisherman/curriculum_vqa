import numpy as np
import matplotlib.pyplot as plt


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