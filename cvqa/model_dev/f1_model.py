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


def get_op(op_seed, op_mat):
    """
    op_seed: [B, d_op]
    op_mat: [d_in, d_out, d_op]
    """
    B, d_op = op_seed.shape
    d_in, d_out, d_op = op_mat.shape
    W = F.linear(op_seed, op_mat.reshape(-1, d_op))  # [B, d_out * d_in]
    return W.reshape(B, d_in, d_out)


def get_shapes(tensors, dims_dict):
    d_reversed = {v: k for k, v in dims_dict.items()}

    def shape(t):
        return list(map(lambda dim: d_reversed.get(dim, dim), t.shape))

    return [shape(t) for t in tensors]


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
            encoder_layers=1,
            encoder_attention_heads=1,
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

        self.m_f1 = F1ModuleMid(args, self.E_c)
        # self.m_f1 = F1ModuleSimple(args, self.E_c)

        d['N_ops'] = 2
        self.E_ops = blocks.Embedding(d['N_ops'], d['w'])

        self.dropout_layer = nn.Dropout(p=0.15)
        self.layer_norm = nn.LayerNorm([d['o']])

    def forward_train(self, sample):
        return self.forward(sample['prompt'], sample['img'])

    def forward_test(self, sample):
        obj_probs = torch.exp(self.forward(sample['prompt'], sample['img']))
        # _, y_pred = torch.max(obj_probs, axis=-1)
        return obj_probs, obj_probs[:, :, 1]

    def forward(self, prompt_tokens, img):

        prompt_encoded, prompt_pad_mask = self.prompt_encoder(prompt_tokens)

        X = img
        X = self.layer_norm(X)
        X = self.dropout_layer(X)
        B, N_objs, _ = X.shape

        op_inputs = self.E_ops.weight  # [N_ops, w]
        op_inputs = op_inputs.repeat(B, 1, 1)  # [B, N_ops, w]

        w_ops = self.op_decoder(op_inputs, prompt_encoded, prompt_pad_mask)  # [B, N_ops, w]
        # w_ops = self.layer_norm(w_ops)
        ops = w_ops
        f1_op = ops[:, 0, :]
        X_w = torch.ones(B, N_objs).to(device)
        X_w_pred = self.m_f1(X, X_w, f1_op)

        class_preds = torch.zeros(B, N_objs, 2).to(device)
        class_preds[:, :, 1] = X_w_pred  # the probability the object is attended
        class_preds[:, :, 0] = 1 - X_w_pred  # the probability object is not attended

        ret = torch.log(class_preds)
        return ret


class F1ModuleSimple(nn.Module):

    def __init__(self, args, E_c):
        super().__init__()
        self.args = args
        d = parse_dims_dict(args)
        self.dims_dict = d

        self.W_concepts = nn.Sequential(
            nn.Linear(d['w'], d['c']),
            # nn.ReLU(),
            # nn.Linear(d['c'], d['c']),
            # nn.ReLU(),
        )

        self.W_viz = nn.Sequential(
            nn.Linear(d['o'], d['o']),
            nn.ReLU(),
            nn.Linear(d['o'], d['o']),
        )

        self.E_c = E_c
        self.layer_norm = nn.LayerNorm([d['o']])

    def forward(self, X, X_weights_in, op):
        """
        X: [B, N_o, d_o]
        X_weights_in: [B, N_o]
        op: [B, d_c]
        """
        B, N_o, _ = X.shape

        X = self.W_viz(X)
        X = self.layer_norm(X)

        P = self.W_concepts(op)

        # 3)
        X_c_logits = torch.matmul(
            X,             # [B, N_o, d_c]
            P.unsqueeze(2)   # [B, d_c, 1]
        ).squeeze(2)
        #  X_c_logits:  [B, N_o]
        EPS = 1e-4
        XP_res = torch.sigmoid(X_c_logits) * (1 - 2*EPS) + EPS

        return X_weights_in * XP_res


class F1ModuleMid(nn.Module):

    def __init__(self, args, E_c):
        super().__init__()
        self.args = args
        d = parse_dims_dict(args)
        self.dims_dict = d

        self.W_viz = nn.Sequential(
            nn.Linear(d['o'], d['o']),
            nn.ReLU(),
            nn.Linear(d['o'], d['o']),
        )
        self.x_norm = nn.LayerNorm(d['o'])

        self.W_ck = nn.Linear(d['c'], d['k'])

        self.CW_concepts = blocks.CondLinear(d['o'], d['c'], d['k'], bias=True)
        self.E_c = E_c

    def forward(self, X, X_weights_in, concept):
        """
        X: [B, N_o, d_o]
        X_weights_in: [B, N_o]
        op: [B, d_c]

        returns [B, N_o, d_o]
        """
        B, N_o, _ = X.shape

        X = self.W_viz(X)
        X = self.x_norm(X)

        concept_cat = self.W_ck(concept)  # [B, d_k]
        # Transform X conditioned on the concept category (eg color)
        X_c = self.CW_concepts(X, concept_cat)  # [B, N_o, d_c]

        X_c_logits = torch.matmul(
            X_c,                    # [B, N_o, d_c]
            concept.unsqueeze(2)    # [B, d_c, 1]
        ).squeeze(2)  # [B, N_o]

        EPS = 1e-4
        XP_res = torch.sigmoid(X_c_logits) * (1 - 2*EPS) + EPS

        return X_weights_in * XP_res