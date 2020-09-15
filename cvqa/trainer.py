import logging
import statistics
import os
import warnings

from datetime import datetime

import torch
import torch.utils as torch_utils
import torch.nn as nn

from torch.utils.tensorboard import SummaryWriter

import fairseq.utils as fairseq_utils

from tqdm import tqdm


class Trainer(object):

    def __init__(self, ignore_index=None, log_dir=None):
        self.ignore_index = ignore_index

        self.is_cuda = False
        if torch.cuda.is_available():
            self.is_cuda = True

        self.log_dir = log_dir

        self.summary_writer = None

    def train(self, model, train_dataset, dev_dataset, optimizer, num_epochs=10, batch_size=32):
        log_dir = self.log_dir
        if log_dir is not None:
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            log_dir = os.path.join(self.log_dir, f'run-{current_time}')

        self.summary_writer = SummaryWriter(log_dir)

        training_generator = torch_utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        dev_generator = torch_utils.data.DataLoader(
            dev_dataset, batch_size=batch_size, shuffle=True
        )

        # try:
        #     sample = next(iter(training_generator))
        #     self.summary_writer.add_graph(model, self._model_input(sample))
        # except:
        #     pass
            # logging.exception("Error writing graph")
            # warnings.warn("deprecated", DeprecationWarning)

        if self.ignore_index is not None:
            ce_crit = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        else:
            ce_crit = nn.CrossEntropyLoss()

        train_loss = []
        train_acc = []
        dev_acc = []

        def eval_step(epoch):
            cur_train_acc = self.evaluate(model, training_generator, iter_lim=100)
            train_acc.append(cur_train_acc)
            cur_dev_acc = self.evaluate(model, dev_generator)
            dev_acc.append(cur_dev_acc)
            self.summary_writer.add_scalar('Accuracy/train', cur_train_acc, epoch)
            self.summary_writer.add_scalar('Accuracy/dev', cur_dev_acc, epoch)

        for epoch in range(num_epochs):
            eval_step(epoch)
            with tqdm(training_generator) as prg_train:
                for i, sample in enumerate(prg_train):

                    info = self.train_step(model, sample, optimizer, ce_crit)

                    train_loss.append(info['loss'])
                    self.summary_writer.add_scalar('Loss/train', info['loss'], epoch * len(training_generator) + i)

                    running_mean_loss = statistics.mean(train_loss[-min(len(train_loss), 100):])
                    status_str = f'[epoch={epoch}, train_acc={train_acc[-1]:.2f}, dev_acc={dev_acc[-1]:.2f}] loss: {running_mean_loss:.3f}'
                    prg_train.set_description(status_str)

        eval_step(num_epochs)

        return train_loss, train_acc, dev_acc

    def _model_input(self, sample):
        raise NotImplementedError()

    def _model_fwd(self, model, sample):
        raise NotImplementedError()

    def train_step(self, model, sample, optimizer, criterion):
        # to make reproducible with checkpoints, set seed here appropriately (see example in LabelSmoothedCrossEntropyCriterion)

        model.train()
        optimizer.zero_grad()
        sample = self.prep_sample(sample)

        logits, targets = self._model_fwd(model, sample)

        loss = criterion(logits, targets.squeeze())

        loss.backward()
        # sample_size = sample['ntokens']
        # optimizer.multiply_grads(1. / float(sample_size))  # Needed?
        optimizer.step()
        #     self.set_num_updates(self.get_num_updates() + 1)
        # fairseq_utils.clear_cuda(self.args, 0)

        logging_output = {
            'loss': loss.data.item(),
        }
        return logging_output

    def prep_sample(self, sample):
        if self.is_cuda:
            return fairseq_utils.move_to_cuda(sample)
        else:
            return sample

    def evaluate(self, model, data_generator, iter_lim=None):
        """
        Evaluate the given model on the given dataset.

        :param model: instance of torch.nn.Module
        :param data_generator: Instance of torch.utils.data.DataLoader
        :param iter_lim: Run only the specified iterations count
        :return: Accuracy; correct/total
        """

        model.eval()
        with torch.set_grad_enabled(False):

            correct = 0
            total = 0

            for i, sample in enumerate(data_generator):
                if iter_lim is not None and i >= iter_lim:
                    break

                sample = self.prep_sample(sample)

                logits, y_true = self._model_fwd(model, sample)

                _, y_pred = torch.max(logits.data, -1)

                if self.ignore_index is not None:
                    mask = y_true.ne(self.ignore_index)
                    y_true = y_true[mask]
                    y_pred = y_pred.unsqueeze(1)[mask]

                correct += (y_pred == y_true).sum()
                total += y_true.size(0)

            return float(correct) / float(total)


class VQATrainer(Trainer):

    def __init__(self, ignore_index=None, log_dir=None):
        super().__init__(ignore_index, log_dir)

    def _model_input(self, sample):
        return sample['prompt'], sample['img'], sample['target']

    def _model_fwd(self, model, sample):
        model_output = model(src_tokens=sample['prompt'], src_img=sample['img'], prev_output_tokens=sample['target'])
        #     model_output: (
        #         0: Tensor[B, No, V], --> real output, aka logits, the unnormalized scores over each token
        #         1: {
        #             attn: Tensor: [B, No, Ni],  --> cross attention
        #             inner_states: list[Tensor: [?, ?, d]]  --> looks like decoder internal layers
        #         }
        #     )
        logits = model_output[0]  # [B, No, V]
        logits = logits.view(-1, logits.size(-1))  # [B*No, V]
        targets = sample['target'].view(-1, 1)  # [B*No]
        return logits, targets


class ImageClassifierTrainer(Trainer):

    def __init__(self, log_dir=None):
        super().__init__(log_dir=log_dir)

    def _model_input(self, sample):
        return sample['prompt'], sample['img']

    def _model_fwd(self, model, sample):
        model_output = model(*self._model_input(sample))
        return model_output, sample['target']