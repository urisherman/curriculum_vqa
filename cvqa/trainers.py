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

    def __init__(self, log_dir=None, ignore_index=None, progressbar='auto', pred_target='target'):
        self.ignore_index = ignore_index

        self.is_cuda = False
        if torch.cuda.is_available():
            self.is_cuda = True

        self.log_dir = log_dir
        self.summary_writer = None
        self.progressbar = progressbar
        self.pred_target = pred_target

    def train(self, model, train_dataset, dev_dataset, optimizer, optim_sched=None, traintracker=None, num_epochs=10, batch_size=32):
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

        if self.ignore_index is not None:
            ce_crit = nn.CrossEntropyLoss(ignore_index=self.ignore_index)
        else:
            ce_crit = nn.CrossEntropyLoss()

        train_loss = []
        train_acc = []
        dev_acc = []

        def eval_epoch(epoch):
            cur_train_acc = self.evaluate(model, training_generator, iter_lim=500)
            train_acc.append(cur_train_acc)
            cur_dev_acc = self.evaluate(model, dev_generator)
            dev_acc.append(cur_dev_acc)
            if self.summary_writer:
                self.summary_writer.add_scalar('Accuracy/train', cur_train_acc, epoch)
                self.summary_writer.add_scalar('Accuracy/dev', cur_dev_acc, epoch)
            if traintracker:
                traintracker.start('epoch')
                traintracker.log_metric('train_acc', cur_train_acc)
                traintracker.log_metric('dev_acc', cur_dev_acc)

        def train_step_and_log(epoch, i, sample, prg_train=None):
            if traintracker:
                traintracker.start('train_step')
            info = self.train_step(model, sample, optimizer, ce_crit)
            if optim_sched is not None:
                optim_sched.step()

            train_loss.append(info['loss'])
            if self.summary_writer:
                self.summary_writer.add_scalar('Loss/train', info['loss'], epoch * len(training_generator) + i)
            if traintracker:
                traintracker.log_metric('loss', info['loss'])

            if i % 10 == 0 and prg_train is not None:
                running_mean_loss = statistics.mean(train_loss[-min(len(train_loss), 100):])
                steps = epoch*batch_size + i
                status_str = f'[epoch={epoch}, steps={steps}, train_acc={train_acc[-1]:.2f}, dev_acc={dev_acc[-1]:.2f}] loss: {running_mean_loss:.3f}'
                prg_train.set_description(status_str)

        def train_epoch(train_data, epoch, prg_train=None):
            for i, sample in enumerate(train_data):
                train_step_and_log(epoch, i, sample, prg_train)

            if self.summary_writer is not None:
                for tag, parm in model.named_parameters():
                    if parm.grad is None:
                        warnings.warn(f'model parameter {tag} has no gradients')
                    self.summary_writer.add_histogram(tag, parm.grad.data.cpu().numpy(), epoch)

        if self.progressbar == 'none':
            for epoch in range(num_epochs):
                eval_epoch(epoch)
                train_epoch(training_generator, epoch)
        elif self.progressbar == 'steps' or (self.progressbar == 'auto' and num_epochs < 5):
            for epoch in range(num_epochs):
                eval_epoch(epoch)
                with tqdm(training_generator) as prg_train:
                    train_epoch(prg_train, epoch, prg_train)
        else:  # self.progressbar == 'epochs'
            with tqdm(range(num_epochs)) as prg_train:
                for epoch in prg_train:
                    eval_epoch(epoch)
                    train_epoch(training_generator, epoch, prg_train)

        eval_epoch(num_epochs)

        if traintracker:
            traintracker.plot('loss', 'train_step')
            traintracker.plot('train_acc', 'epoch')
            traintracker.plot('dev_acc', 'epoch')

        return train_loss, train_acc, dev_acc
    #
    # def _model_input(self, sample):
    #     return sample['prompt'], sample.get('img'), sample['target_attention_mask']

    def train_step(self, model, sample, optimizer, criterion):
        # to make reproducible with checkpoints, set seed here appropriately (see example in LabelSmoothedCrossEntropyCriterion)

        model.train()
        optimizer.zero_grad()
        sample = self.prep_sample(sample)

        logits = self.forward_train(model, sample)
        logits = logits.flatten(end_dim=1)  # [B, No * V_target]
        targets = sample[self.pred_target].flatten()  # [B * No]

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

    def forward_train(self, model, sample):
        model_fn = getattr(model, "forward_train", None)
        if callable(model_fn):
            return model.forward_train(sample)
        else:
            return model.forward(sample['prompt'], sample['img'])

    def forward_test(self, model, sample):
        model_fn = getattr(model, "forward_test", None)
        if callable(model_fn):
            _, y_pred = model.forward_test(sample)
            return y_pred
        else:
            logits = model.forward(sample['prompt'], sample['img'])
            _, y_pred = torch.max(logits, axis=-1)
            return y_pred

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

                y_pred = self.forward_test(model, sample)
                y_pred = y_pred.flatten()  # [B*No]
                y_true = sample[self.pred_target].flatten()  # [B*No]

                if self.ignore_index is not None:
                    mask = y_true.ne(self.ignore_index)
                    y_true = y_true[mask]
                    y_pred = y_pred[mask]

                correct += (y_pred == y_true).sum()
                total += y_true.size(0)

            return float(correct) / float(total)

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
                y_pred = self.forward_test(model, s)

                y_trues[s['index']] = targets
                y_preds[s['index']] = y_pred

        return y_trues.cpu().numpy(), y_preds.cpu().numpy()


