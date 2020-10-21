import torch
import torch.nn as nn
import torch.nn.functional as F

from cvqa.models2 import Embedding


class StructuredImageModel(nn.Module):

    def __init__(self, d_obj):
        super().__init__()
        # vocab = dataset.struct_viz_vocab
        # self.rep_vocab = vocab
        # d_num = dataset.samples[0]['viz_rep']['numerics_img'].shape[1]
        # self.rep_tokens_embedding = Embedding(len(vocab), d_obj-d_num)

        # self.bn_layer = nn.BatchNorm1d(d_obj)
        self.layer_norm = nn.LayerNorm([d_obj])

    def forward(self, structured_rep):
        # img_tokens = structured_rep['tokens']  # [B, N_objs, N_tokens]
        # rep_embed = self.rep_tokens_embedding(img_tokens)  # [B, N_objs, N_tokens, d_embed]
        # rep_embed = torch.sum(rep_embed, dim=2)  # [B, N_objs, d_embed]
        #
        # img_numerics = structured_rep['numerics']  # [B, N_objs, d_num]
        #
        # img = torch.cat([rep_embed, img_numerics], dim=2)  # [B, N_objs, d_embed + d_num]
        # B, N_objs, d_o = img.shape
        # img = img.reshape(-1, d_o)
        # img = self.bn_layer(img)
        # img = img.reshape(B, N_objs, d_o)

        # img = self.bn_layer(structured_rep)
        img = self.layer_norm(structured_rep)

        return img