import pprint

import numpy as np
import matplotlib.pyplot as plt
from textwrap import wrap
import pandas as pd
import torch
from sklearn import metrics
import seaborn as sn

from cvqa import utils


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

    flip_viz_rep = False
    if dataset.use_viz_rep:
        flip_viz_rep = True
        dataset.use_viz_rep = False

    for i in range(k):
        idx = np.random.randint(0, len(dataset))
        sample = dataset[idx]
        sample['text_prompt'] = dataset.samples[idx]['prompt']
        sample['text_target'] = dataset.samples[idx]['target']
        imshow(sample['img'],
               title=sample['text_prompt'] + ' "' + sample['text_target'] + '"',
               ax=axs[i])

    if flip_viz_rep:
        dataset.use_viz_rep = True


def plot_conf_mat(y_true, y_pred, labels):
    conf_mat = metrics.confusion_matrix(y_true, y_pred)
    conf_df = pd.DataFrame(conf_mat, index=labels, columns=labels)

    x_ax_length = len(labels)*1.2
    fig, ax = plt.subplots(figsize = (x_ax_length, x_ax_length*.8))
    hm = sn.heatmap(conf_df, annot=True, annot_kws={'fontsize': 16}, fmt='g', ax=ax)
    hm.set_xticklabels(hm.get_xmajorticklabels(), fontsize=12)
    hm.set_yticklabels(hm.get_ymajorticklabels(), fontsize=12)

    ax.set_xlabel('Pred', fontsize=18)
    ax.set_ylabel('True', fontsize=18)


def one_word_conf_mat(y_true, y_pred, vocab):
    y_true_list = list(y_true.squeeze())
    y_pred_list = list(y_pred.squeeze())

    labels = set(y_true_list)
    labels.update(y_pred_list)
    labels = list(labels)
    labels.sort()
    label_to_idx = {l: i for i, l in enumerate(labels)}

    y_true_cls = list(map(lambda x: label_to_idx[x], y_true_list))
    y_pred_cls = list(map(lambda x: label_to_idx[x], y_pred_list))

    str_labels = list(map(lambda l: vocab.string([l]), label_to_idx))

    plot_conf_mat(y_true_cls, y_pred_cls, str_labels)


def test_clf_sample(model, dataset, sample_idx=None):
    if sample_idx is None:
        sample_idx = np.random.randint(len(dataset))

    sample = dataset[sample_idx]
    sample['text_prompt'] = dataset.samples[sample_idx]['prompt']
    sample['text_target'] = dataset.samples[sample_idx]['target']

    sample['img'] = sample['img'].unsqueeze(0)
    sample['prompt'] = torch.tensor([sample['prompt']])
    sample['target'] = torch.tensor([sample['target']])
    sample = utils.sample_to_cuda(sample)

    logits = model(sample['prompt'], sample['img'])
    _, y_pred = torch.max(logits, -1)
    y_pred = y_pred.cpu().numpy()[0]

    print(sample['text_prompt'])
    print('True: ' + sample['text_target'])
    print('Pred: ' + dataset.idx_to_cls[y_pred])
    imshow(sample['img'][0].cpu())


def test_natural_sample(model, dataset, sample_idx=None):
    if sample_idx is None:
        sample_idx = np.random.randint(len(dataset))

    sample = dataset[sample_idx]
    sample['text_prompt'] = dataset.samples[sample_idx]['prompt']
    sample['text_target'] = dataset.samples[sample_idx]['target']

    if type(sample['prompt']) == int:
        sample['prompt'] = torch.tensor([sample['prompt']])
    if type(sample['target']) == int:
        sample['target'] = torch.tensor([sample['target']])

    sample['img'] = sample['img'].unsqueeze(0)
    sample['prompt'] = sample['prompt'].unsqueeze(0)
    sample['target'] = sample['target'].unsqueeze(0)

    sample = utils.sample_to_cuda(sample)

    # if dataset.teacher_forcing:
    #     logits = model.forward_predict(sample['prompt'], sample['img'])
    # else:
    logits = model(sample['prompt'], sample['img'], None)

    _, y_pred = torch.max(logits, -1)
    y_pred = y_pred.cpu().numpy()[0]

    print(f'Sample index: {sample_idx}')
    print('Prompt: ' + sample['text_prompt'])
    print('Decoded encoded prompt: ' + dataset.vocab.string(sample['prompt']))
    print('True: ' + sample['text_target'])
    print('Pred: ' + dataset.vocab.string(y_pred))
    if dataset.use_viz_rep:
        # pp = pprint.PrettyPrinter(indent=4)
        # pp.pprint(sample['img'][0].cpu())
        print('* Encoded Structured Img Rep: ' + str(sample['img'][0].cpu()))
        imshow(dataset.load_img(sample_idx).cpu())
    else:
        imshow(sample['img'][0].cpu())