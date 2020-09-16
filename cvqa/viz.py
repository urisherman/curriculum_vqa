import numpy as np
import matplotlib.pyplot as plt
from textwrap import wrap


def plot_training(train_loss, train_acc, dev_acc):

    fig, axs = plt.subplots(nrows=2, figsize=(15, 6))

    axs[0].plot(train_loss, '-', label='Train Loss');
    axs[0].legend()
    axs[1].plot(train_acc, '-o', label='Train Acc');
    axs[1].plot(dev_acc, '-o', label='Val Acc');
    axs[1].legend()

    plt.tight_layout()


def imshow(inp, title=None, ax=None):
    """Imshow for Tensor."""
    inp = inp.numpy().transpose((1, 2, 0))
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)

    if ax is None:
        fig, ax = plt.subplots()

    ax.imshow(inp)
    if title is not None:
        ax.set_title("\n".join(wrap(title, 32)))
    # plt.pause(0.001)  # pause a bit so that plots are updated


def show_samples(dataset, k=4):
    fig, axs = plt.subplots(1, k, figsize=(16, 4))

    for i in range(k):
        idx = np.random.randint(0, len(dataset))
        sample = dataset[idx]
        sample['text_prompt'] = dataset.samples[idx]['prompt']
        sample['text_target'] = dataset.samples[idx]['target']
        imshow(sample['img'],
               title=sample['text_prompt'] + ' "' + sample['text_target'] + '"',
               ax=axs[i])

