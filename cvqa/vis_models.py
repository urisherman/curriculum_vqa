import torch
import torch.nn as nn
import torch.nn.functional as F

from cvqa.models2 import Embedding


class StructuredImageModel(nn.Module):

    def __init__(self, rep_vocab, d_obj):
        super().__init__()
        self.rep_vocab = rep_vocab
        self.rep_tokens_embedding = Embedding(len(rep_vocab), d_obj-3)

    def forward(self, structured_rep):
        img_rep_tokens = structured_rep['tokens']
        rep_embed = self.rep_tokens_embedding(img_rep_tokens)  # [B, N_tokens, d_embed]

        rep_embed = torch.sum(rep_embed, dim=1)  # [B, d_embed]

        locsize = structured_rep['locsize']  # [B, 3]

        img = torch.cat([rep_embed, locsize], dim=1)

        return img.unsqueeze(1)