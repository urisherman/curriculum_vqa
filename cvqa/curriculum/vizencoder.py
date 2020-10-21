import itertools
import math
import random
from tqdm import tqdm

import torch
import torch.nn as nn
import torch.nn.functional as F

from cvqa.model_dev.utils import Vocabulary


class AutoencConceptsModel(nn.Module):

    def __init__(self, concept_dict, d_img, N_c, h=10):
        super().__init__()
        N_k = len(concept_dict)

        self.E = nn.Embedding(N_c, h)

        self.enc = nn.Linear(N_k * h, d_img)
        self.dec = nn.Parameter(torch.Tensor(N_k, d_img, h))
        nn.init.kaiming_uniform_(self.dec, a=math.sqrt(5))

        nn.init.normal_(self.E.weight, mean=0, std=h ** -0.5)

    def encode_img_tokens(self, X):
        """
        X: [B, N_k]
        """
        B, N_k = X.shape
        X_vecs = self.E(X)
        X_vecs = X_vecs.reshape(B, -1)
        img_rep = self.enc(X_vecs)
        return img_rep

    def forward(self, X):
        """
        X: [B, N_k]
        """
        img_rep = self.encode_img_tokens(X)

        img_h_vecs = torch.matmul(
            img_rep,  # [B, d_img]
            self.dec  # [N_k, d_img, h]
        ).transpose(0, 1)  # [B, N_k, h]

        logits = F.linear(img_h_vecs, self.E.weight)  # [B, N_k, N_c]
        return logits


def eval_model(model, ds):
    total = 0
    correct = 0
    for i, X in enumerate(torch.utils.data.DataLoader(ds, batch_size=32, shuffle=False)):
        X = X[0]
        logits = model(X)
        logits = torch.flatten(logits, start_dim=0, end_dim=1)
        y = torch.flatten(X, start_dim=0)
        _, y_pred = torch.max(logits, dim=-1)
        correct += torch.sum(y_pred == y).item()
        total += len(y)
    return float(correct) / total


def train_model(model, data_loader, num_epochs=100):
    model.train()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = nn.CrossEntropyLoss()

    losses = []
    with tqdm(range(num_epochs)) as epochsbar:
        for e in (epochsbar):
            for i, X in enumerate(data_loader):
                X = X[0]
                logits = model(X)
                logits = torch.flatten(logits, start_dim=0, end_dim=1)
                y = torch.flatten(X, start_dim=0)

                optimizer.zero_grad()
                loss = loss_fn(logits, y)
                loss.backward()
                optimizer.step()

                losses.append(loss.item())
                epochsbar.set_description(f'Img Encoder: Loss={loss.item():.3f}')

    model.eval()
    for param in model.parameters():
        param.requires_grad = False
    return losses


class VizEncoder(object):

    def __init__(self, concept_dict, numeric_fields=None, d_img=10, N_samples=6000, num_epochs=10):
        self.concept_dict = concept_dict
        self.numeric_fields = numeric_fields

        concept_keys = list(concept_dict.keys())
        concept_values = list(itertools.chain(*concept_dict.values()))

        concepts_vocab = Vocabulary()
        for token in itertools.chain(concept_keys, concept_values):
            concepts_vocab.encode_symbol(token)

        N_c = len(concepts_vocab)

        concepts_vocab.build()

        # d_words = 100
        #
        # pretrained_embedding = GloVe(name='6B', dim=d_words, is_include=lambda w: w in vocab_set)
        # embedding_weights = torch.Tensor(N_c, pretrained_embedding.dim)
        # for i, token in enumerate(encoder.vocab):
        #     embedding_weights[i] = pretrained_embedding[token]

        objects = []
        for i in range(N_samples):
            obj = []
            for ck in concept_dict:
                cv = random.choice(concept_dict[ck])
                c_id = concepts_vocab.encode_symbol(cv)
                obj.append(c_id)
            objects.append(torch.tensor(obj))

        X_all = torch.stack(objects)
        # y_true = torch.stack(objects)
        # X_all = embedding_weights[y_true]

        # Train model
        model = AutoencConceptsModel(concept_dict, d_img, N_c)

        ds = torch.utils.data.TensorDataset(X_all)
        data_loader = torch.utils.data.DataLoader(ds, batch_size=32, shuffle=True)

        losses = train_model(model, data_loader, num_epochs=num_epochs)
        acc = eval_model(model, ds)

        self.tokens_vocab = concepts_vocab
        self.X_all = X_all
        self.model = model
        self.losses = losses
        self.acc = acc

    def encode(self, viz):
        viz_encoded_tokens = torch.zeros(len(viz['objects']), len(self.concept_dict))
        # total_numerics = map(sum, self.numeric_fields.values())
        viz_numerics = []  #torch.zeros(len(viz['objects']), total_numerics)

        for i, o in enumerate(viz['objects']):
            obj_tokens = []
            for ck in self.concept_dict:
                obj_tokens.append(self.tokens_vocab.encode_symbol(o[ck]))
            viz_encoded_tokens[i] = torch.tensor(obj_tokens)

            num_vec = []
            if self.numeric_fields is not None:
                for nf in self.numeric_fields:
                    num_val = o[nf]
                    if type(num_val) == list:
                        num_vec += num_val
                    else:
                        num_vec.append(num_val)
            viz_numerics.append(torch.Tensor(num_vec))

        viz_numerics = torch.stack(viz_numerics)
        viz_numerics = [viz_numerics] * 3
        tokens_enc = self.model.encode_img_tokens(viz_encoded_tokens.long())

        encoded = torch.cat([tokens_enc] + viz_numerics, 1)
        noise = torch.randn_like(encoded) * .05
        return encoded + noise


if __name__ == '__main__':
    concept_dict = {
        'color': ['blue', 'brown', 'cyan', 'gray', 'green', 'purple', 'red', 'yellow'],
        'material': ['metal', 'rubber', 'wood', 'plastic'],
        'shape': ['triangle', 'circle', 'square', 'polytope']
    }

    vizenc = VizEncoder(concept_dict, numeric_fields=['size', 'location'], d_img=5)
    z = vizenc.encode({
        'objects': [{
            'location': [0.3, 0.76],
            'size': 0.15,
            'color': 'yellow',
            'material': 'metal',
            'shape': 'polytope'
    }, {
            'location': [0.4, 0.59],
            'size': 0.08,
            'color': 'brown',
            'material': 'rubber',
            'shape': 'polytope'},
    ]})
    print(z)