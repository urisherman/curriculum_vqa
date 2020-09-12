import statistics

import torch
import torch.utils as torch_utils
import torch.nn as nn

import fairseq.utils as fairseq_utils

from tqdm import tqdm


class Trainer:

    def __init__(self, ignore_index):
        self.ignore_index = ignore_index

        self.is_cuda = False
        if torch.cuda.is_available():
            self.is_cuda = True

    def train(self, model, train_dataset, optimizer, num_epochs=10, batch_size=32):
        training_generator = torch_utils.data.DataLoader(
            train_dataset, batch_size=batch_size, shuffle=True
        )

        ce_crit = nn.CrossEntropyLoss(ignore_index=self.ignore_index)

        train_loss = []
        train_acc = []
        dev_acc = []

        for epoch in range(num_epochs):
            cur_train_acc = self.evaluate(model, training_generator, iter_lim=100)
            train_acc.append(cur_train_acc)

            with tqdm(training_generator) as prg_train:
                for i, sample in enumerate(prg_train):
                    # sample = {
                    #             'id': Tensor[B],
                    #             'nsentences': B,
                    #             'ntokens': ?,
                    #             'net_input': {src_tokens, src_lengths, prev_output_tokens},
                    #             'target': Tensor[B, No]
                    #         }

                    info = self.train_step(model, sample, optimizer, ce_crit)

                    train_loss.append(info['loss'])
                    running_mean_loss = statistics.mean(train_loss[-min(len(train_loss), 100):])
                    status_str = f'[{epoch}] loss: {running_mean_loss:.3f}'
                    prg_train.set_description(status_str)

        cur_train_acc = self.evaluate(model, training_generator, iter_lim=100)
        train_acc.append(cur_train_acc)
        return train_loss, train_acc, dev_acc

    def model_fwd(self, model, sample):
        model_output = model(src_tokens=sample['X'], src_img=sample['img'], prev_output_tokens=sample['target'])
        logits = model_output[0]  # [B, No, V]
        logits = logits.view(-1, logits.size(-1))  # [B*No, V]
        targets = sample['target'].view(-1, 1)  # [B*No]
        return logits, targets

    def train_step(self, model, sample, optimizer, criterion):
        # to make reproducible with checkpoints, set seed here appropriately (see example in LabelSmoothedCrossEntropyCriterion)

        model.train()
        optimizer.zero_grad()
        sample = self.prep_sample(sample)

        #     model_output: (
        #         0: Tensor[B, No, V], --> real output, aka logits, the unnormalized scores over each token
        #         1: {
        #             attn: Tensor: [B, No, Ni],  --> cross attention
        #             inner_states: list[Tensor: [?, ?, d]]  --> looks like decoder internal layers
        #         }
        #     )

        logits, targets = self.model_fwd(model, sample)

        loss = criterion(logits, targets.squeeze())

        loss.backward()
        # sample_size = sample['ntokens']
        # optimizer.multiply_grads(1. / float(sample_size))  # Needed?
        optimizer.step()
        #     self.set_num_updates(self.get_num_updates() + 1)
        # fairseq_utils.clear_cuda(self.args, 0)

        logging_output = {
            'loss': loss.data.item(),
            # 'sample_size': sample_size,
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

                logits, y_true = self.model_fwd(model, sample)

                _, y_pred = torch.max(logits.data, -1)

                mask = y_true.ne(self.ignore_index)
                y_true = y_true[mask]
                y_pred = y_pred.unsqueeze(1)[mask]

                correct += (y_pred == y_true).sum()
                total += y_true.size(0)

            return float(correct) / float(total)