import torch
import math

from torch import nn
import torch.nn.functional as F

from cvqa.model_dev import blocks

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def nn_parameter(*shape):
    W = nn.Parameter(torch.Tensor(*shape))
    nn.init.kaiming_uniform_(W, a=math.sqrt(5))
    return W


def default_args():
    return {
        'd_a': 4,
        'd_w': 16,
        'd_c': 16,
        'd_o': 16,
        'd_k': 8,
    }


def parse_dims_dict(args):
    return {
        'a': args['d_a'],
        'w': args['d_w'],
        'c': args['d_c'],
        'o': args['d_o'],
        'k': args['d_k'],
    }


USE_CONCEPT_SPACE = False


class ParentModel(nn.Module):

    def __init__(self, prompt_vocab, answer_vocab, args):
        super().__init__()
        self.args = args

        d = parse_dims_dict(args)
        self.dims_dict = d
        self.prompt_vocab = prompt_vocab
        self.answer_vocab = answer_vocab

        self.prompt_encoder = blocks.TransformerEncoder.build(
            prompt_vocab,
            d['w'],
            encoder_layers=2,
            encoder_attention_heads=2,
            encoder_dropout=0
        )

        self.op_decoder = blocks.TransformerDecoder.build(
            d['w'],
            decoder_ffn_dim=32,
            decoder_layers=1,
            decoder_attention_heads=1,
            decoder_dropout=0)

        d['N_c'] = len(prompt_vocab)
        self.E_c = self.prompt_encoder.tokens_embedding
        self.W_w_op = None

        self.m_ans = AnswerModule(d, answer_vocab)

        d['N_ops'] = 2
        self.E_ops = blocks.Embedding(d['N_ops'], d['w'])

        self.dropout_layer = nn.Dropout(p=0.15)
        self.layer_norm = nn.LayerNorm([d['o']])

    def forward_train(self, sample):
        return self.forward(sample['prompt'], sample['img'], sample['target_attention_mask'])

    def forward_test(self, sample):
        logits = self.forward(sample['prompt'], sample['img'], sample['target_attention_mask'])
        _, y_pred = torch.max(logits, axis=-1)
        return logits, y_pred

    def forward(self, prompt_tokens, img, target_attention_mask=None):

        prompt_encoded, prompt_pad_mask = self.prompt_encoder(prompt_tokens)

        X = img
        X = self.layer_norm(X)
        # X = self.dropout_layer(X)
        B, N_objs, _ = X.shape

        op_inputs = self.E_ops.weight  # [N_ops, w]
        op_inputs = op_inputs.repeat(B, 1, 1)  # [B, N_ops, w]

        w_ops = self.op_decoder(op_inputs, prompt_encoded, prompt_pad_mask)  # [B, N_ops, w]
        # w_ops = self.layer_norm(w_ops)
        ops = w_ops
        ans_op = ops[:, 0, :]
        # ans_op = self.dropout_layer(ans_op)

        X_w = target_attention_mask.float()
        X_w[X_w == -1] = 0
        logits = self.m_ans(X, X_w, ans_op)
        return logits.unsqueeze(1)


class AnswerModule(nn.Module):

    def __init__(self, dims_dict, vocab):
        super().__init__()
        d = dims_dict

        self.vocab = vocab
        self.dims_dict = dims_dict

        self.W_viz = nn.Sequential(
            nn.Linear(d['o'], d['o']),
            nn.ReLU(),
            nn.Linear(d['o'], d['o']),
            nn.ReLU(),
        )

        self.CW_ans = blocks.CondLinear(d['o']+1, d['a'], d['c'], bias=True)
        # self.CW_ans = blocks.CondFFN(d['o'] + 1, 64, d['a'], d['c'], has_bias=True, n_hidden_layers=1)

        self.E_a = blocks.Embedding(len(vocab), d['a'])
        self.x_norm = nn.LayerNorm(d['o'])

    def forward(self, X, X_weights_in, question_op):
        """
        X: [B, N_o, d_o]
        X_weights_in: [B, N_o]
        question_op: [B, d_c]
        """
        # weighted_X = X.sum(axis=1)  # [B, o]
        B, N_o, d_o = X.shape

        X = self.W_viz(X)
        X = self.x_norm(X)

        indicators = torch.ones(B, N_o, 1).to(device)
        if self.training:
            indicators -= torch.rand(B, N_o, 1).to(device) * 0.01

        X = torch.cat([indicators, X], dim=2)
        weighted_X = (X_weights_in.unsqueeze(2) * X).sum(axis=1)  # [B, o+1]
        # weighted_X = (X_weights_in.unsqueeze(2) * torch.ones_like(X)).sum(axis=1)  # [B, o]

        ans_vec = self.CW_ans(weighted_X, question_op)  # [B, a]
        logits = F.linear(ans_vec.squeeze(1), self.E_a.weight)  # [B, N_a]
        return logits


class AttentionAnswerModule(nn.Module):

    def __init__(self, dims_dict, vocab):
        super().__init__()
        d = dims_dict

        self.vocab = vocab
        self.dims_dict = dims_dict

        self.E_a = blocks.Embedding(len(vocab), d['a'])
        self.ATT_ans = blocks.SoftlyMaskedAttention(d['c'], d['o'], d['a'], 64)

    def forward(self, X, X_weights_in, question_op):
        """
        X: [B, N_o, d_o]
        X_weights_in: [B, N_o]
        question_op: [B, d_c]
        """

        B, N_o, d_o = X.shape
        X_aug = torch.cat([X, torch.zeros(B, 1, d_o, dtype=torch.float)], dim=1)
        X_w_aug = torch.cat([X_weights_in, torch.ones(B, 1, dtype=torch.float)], dim=1)
        ans_vec = self.ATT_ans(question_op.unsqueeze(1), X_aug, X_w_aug)  # [B, 1, d_a]
        logits = F.linear(ans_vec.squeeze(1), self.E_a.weight)  # [B, N_a]

        return logits
