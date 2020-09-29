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

from cvqa import datasets, utils

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class VQATrainer(object):

    def __init__(self, log_dir=None, ignore_index=None, sample_mode='natural', progressbar='auto'):
        self.ignore_index = ignore_index

        self.is_cuda = False
        if torch.cuda.is_available():
            self.is_cuda = True

        self.log_dir = log_dir
        self.summary_writer = None
        self.sample_mode = sample_mode
        self.progressbar = progressbar

    def train(self, model, train_dataset, dev_dataset, optimizer, optim_sched=None, num_epochs=10, batch_size=32):
        log_dir = self.log_dir
        if log_dir is not None:
            current_time = datetime.now().strftime('%b%d_%H-%M-%S')
            log_dir = os.path.join(self.log_dir, f'run-{current_time}')
            self.summary_writer = SummaryWriter(log_dir)

        if self.is_cuda:
            model.to(device)

        train_dataset.teacher_forcing = True
        dev_dataset.teacher_forcing = True

        training_generator = torch_utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        dev_generator = torch_utils.data.DataLoader(
            dev_dataset, batch_size=batch_size, shuffle=True
        )

        if self.summary_writer and False:
            try:
                sample = next(iter(training_generator))
                sample = self.prep_sample(sample)
                self.summary_writer.add_graph(model, self._model_input(sample))
            except:
                pass
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
            cur_train_acc = self.evaluate(model, training_generator, iter_lim=500)
            train_acc.append(cur_train_acc)
            cur_dev_acc = self.evaluate(model, dev_generator)
            dev_acc.append(cur_dev_acc)
            if self.summary_writer:
                self.summary_writer.add_scalar('Accuracy/train', cur_train_acc, epoch)
                self.summary_writer.add_scalar('Accuracy/dev', cur_dev_acc, epoch)

        def train_step_and_log(epoch, i, sample, prg_train=None):
            info = self.train_step(model, sample, optimizer, ce_crit)
            if optim_sched is not None:
                optim_sched.step()

            train_loss.append(info['loss'])
            if self.summary_writer:
                self.summary_writer.add_scalar('Loss/train', info['loss'], epoch * len(training_generator) + i)

            if prg_train is not None:
                running_mean_loss = statistics.mean(train_loss[-min(len(train_loss), 100):])
                steps = epoch*batch_size + i
                status_str = f'[epoch={epoch}, steps={steps}, train_acc={train_acc[-1]:.2f}, dev_acc={dev_acc[-1]:.2f}] loss: {running_mean_loss:.3f}'
                prg_train.set_description(status_str)

        if self.progressbar == 'none':
            for epoch in range(num_epochs):
                eval_step(epoch)
                for i, sample in enumerate(training_generator):
                    train_step_and_log(epoch, i, sample)
        elif self.progressbar == 'steps' or (self.progressbar == 'auto' and num_epochs < 20):
            for epoch in range(num_epochs):
                eval_step(epoch)
                with tqdm(training_generator) as prg_train:
                    for i, sample in enumerate(prg_train):
                        train_step_and_log(epoch, i, sample, prg_train)
        else:  # self.progressbar == 'epochs'
            with tqdm(range(num_epochs)) as prg_train:
                for epoch in prg_train:
                    eval_step(epoch)
                    for i, sample in enumerate(training_generator):
                        train_step_and_log(epoch, i, sample, prg_train)

        eval_step(num_epochs)

        return train_loss, train_acc, dev_acc

    def _model_input(self, sample):
        return sample['prompt'], sample.get('img')  # , sample['target']

    def _model_fwd(self, model, sample):
        logits = model(*self._model_input(sample))
        targets = sample['target']

        if self.sample_mode == 'natural':
            # logits.shape == [B, No, V]
            logits = logits.view(-1, logits.size(-1))  # [B*No, V]
            targets = targets.flatten()  # [B*No]
        # else:
        #     logits.shape == [B, L]
        #     targets.shape == [B]

        return logits, targets

    def train_step(self, model, sample, optimizer, criterion):
        # to make reproducible with checkpoints, set seed here appropriately (see example in LabelSmoothedCrossEntropyCriterion)

        model.train()
        optimizer.zero_grad()
        sample = self.prep_sample(sample)

        logits, targets = self._model_fwd(model, sample)

        loss = criterion(logits, targets)

        loss.backward()
        optimizer.step()

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
                    y_pred = y_pred[mask]

                correct += (y_pred == y_true).sum()
                total += y_true.size(0)

            return float(correct) / float(total)

    def get_clf_predictions(self, model, dataset):
        res = utils.torch_zeros(len(dataset), 3, dtype=torch.int64)
        model.eval()
        dataset.teacher_forcing = False

        dloader = torch.utils.data.DataLoader(
            datasets.WithIndicesDataset(dataset), shuffle=False, batch_size=64)

        with torch.set_grad_enabled(False):
            for s in dloader:
                s = self.prep_sample(s)

                logits, y_true = self._model_fwd(model, s)

                _, y_pred = torch.max(logits.data, -1)
                res[s['index']] = torch.stack([s['index'], y_true, y_pred], dim=1)
        return res.cpu()

    def get_predictions(self, model, dataset):
        model.eval()
        dataset.teacher_forcing = False

        dloader = torch.utils.data.DataLoader(
            datasets.WithIndicesDataset(dataset), shuffle=False, batch_size=64)

        y_trues = torch.ones(len(dataset), dataset.N_target, dtype=torch.int64) * -1
        y_preds = torch.ones(len(dataset), dataset.N_target, dtype=torch.int64) * -1
        if utils.IS_CUDA:
            y_trues = y_trues.cuda()
            y_preds = y_preds.cuda()

        with torch.set_grad_enabled(False):
            for s in dloader:
                s = utils.sample_to_cuda(s)

                targets = s['target']
                # TODO: Sort out how we do prediction without prev_output_tokens
                logits = model.forward(*self._model_input(s))

                B, N_out, V = logits.shape
                flat_logits = logits.view(-1, logits.size(-1))  # [B*No, V]
                _, y_pred = torch.max(flat_logits.data, -1)
                y_pred = y_pred.reshape(B, N_out)

                y_trues[s['index']] = targets
                y_preds[s['index']] = y_pred

        return y_trues.cpu().numpy(), y_preds.cpu().numpy()


class ImageClassifierTrainer(VQATrainer):

    def __init__(self, log_dir=None):
        super().__init__(log_dir=log_dir)

    def _model_input(self, sample):
        return sample['prompt'], sample['img']

    def _model_fwd(self, model, sample):
        model_output = model(*self._model_input(sample))
        return model_output, sample['target']


