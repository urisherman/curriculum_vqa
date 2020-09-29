import torch
import torchvision as tv
import math

from torch import nn
import torch.nn.functional as F

from cvqa import fairseq_misc, utils
from cvqa.datasets import LabelIndexer

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class VQAModelV1(nn.Module):

    @staticmethod
    def build(vocab, d=16, img_output_features=20):
        tokens_embed = fairseq_misc.build_embedding(vocab, d)
        encoder = fairseq_misc.build_vqa_encoder(vocab, tokens_embed)
        decoder = fairseq_misc.build_decoder(vocab, tokens_embed)
        img_perceptor = BasicImgModel(d, img_output_features)
        return VQAModelV1(encoder, img_perceptor, decoder, bos=vocab.bos_index)

    @staticmethod
    def struct_img_build(dataset, d=16, img_output_features=20):
        vocab = dataset.vocab
        tokens_embed = fairseq_misc.build_embedding(vocab, d)
        encoder = fairseq_misc.build_vqa_encoder(vocab, tokens_embed)
        decoder = fairseq_misc.build_decoder(vocab, tokens_embed)
        img_perceptor = StructuredImageModel(dataset.struct_viz_vocab, d, img_output_features)
        return VQAModelV1(encoder, img_perceptor, decoder, bos=vocab.bos_index)

    def __init__(self, vqa_encoder, img_perceptor, decoder, bos=0):
        super().__init__()
        self.vqa_encoder = vqa_encoder
        self.img_perceptor = img_perceptor
        self.decoder = decoder
        self.bos = bos

    def forward(self, prompt_tokens, img, prev_output_tokens=None):
        img_embedding = self.img_perceptor(img)
        encoder_out = self.vqa_encoder(prompt_tokens, img_embedding)

        B, N_in = prompt_tokens.shape
        bos_tensor = torch.ones(B, 1, dtype=torch.int64).to(device) * self.bos
        decoder_out = self.decoder(bos_tensor, encoder_out=encoder_out)

        return decoder_out[0]


class StructuredImageModel(nn.Module):

    def __init__(self, rep_vocab, output_dim, output_channels=None):
        super().__init__()
        self.rep_vocab = rep_vocab
        self.rep_tokens_embedding = Embedding(len(rep_vocab), output_dim)
        self.output_channels = output_channels

    def forward(self, img_rep_tokens):
        rep_embed = self.rep_tokens_embedding(img_rep_tokens)
        B, F_img, d = rep_embed.shape

        if self.output_channels > F_img:
            pad_length = self.output_channels - F_img
            rep_embed = F.pad(rep_embed, (0, 0, 0, pad_length), value=self.rep_vocab.pad_index)

        return rep_embed


class BasicImgModel(nn.Module):

    def __init__(self, output_dim, output_channels=None, pretrained=True):
        super().__init__()

        self.backbone = tv.models.resnet18(pretrained=pretrained)
        backbone_output = self.__backbone_forward(torch.rand(1, 3, 224, 224))
        B, C, H, W = backbone_output.shape

        if output_channels is not None:
            self.stem_conv = nn.Conv2d(C, output_channels, kernel_size=1, padding=0, bias=False)
            nn.init.kaiming_normal_(self.stem_conv.weight, mode='fan_out', nonlinearity='relu')
            self.fc = nn.Linear(H * W, output_dim)
        else:
            self.stem_conv = None
            self.fc = nn.Linear(C * H * W, output_dim)

    def __backbone_forward(self, img_sample):
        bb = self.backbone
        x = img_sample
        x = bb.conv1(x)
        x = bb.bn1(x)
        x = bb.relu(x)
        x = bb.maxpool(x)

        x = bb.layer1(x)
        x = bb.layer2(x)
        x = bb.layer3(x)

        return x

    def forward(self, img):
        bb_output = self.__backbone_forward(img)

        if self.stem_conv is not None:
            bb_output = self.stem_conv(bb_output)
            bb_output = torch.flatten(bb_output, start_dim=2)
        else:
            bb_output = torch.flatten(bb_output, start_dim=1)

        output = self.fc(bb_output)
        return output


class VQAPromptOpModel(nn.Module):

    @staticmethod
    def build(d, dataset, c=None, img_perceptor=None):
        prompt_embeddings, target_embeddings = build_embeddings(d, dataset, c=c)
        return VQAPromptOpModel(prompt_embeddings, target_embeddings, img_perceptor=img_perceptor)

    def __init__(self, prompt_embedding, target_embedding, img_perceptor=None):
        super().__init__()

        if img_perceptor is None:
            img_perceptor = BasicImgModel(20)
        self.img_perceptor = img_perceptor
        img_embedding = img_perceptor(torch.rand(1, 3, 224, 224))
        B, P = img_embedding.shape

        dims = {
            'P': P,  # perception embedding dim
            'V': prompt_embedding.num_embeddings,  # num of prompt tokens
            'd': prompt_embedding.embedding_dim,  # prompt tokens embedding
            'L': target_embedding.num_embeddings,  # num of target toekns
            'c': target_embedding.embedding_dim  # target tokens embedding
        }
        self.dims = dims
        self.prompt_embedding = prompt_embedding
        self.target_embedding = target_embedding

        # The operators operator
        # Given an embedded prompt, output a P --> c operator
        self.W_op = nn.Parameter(torch.Tensor(dims['d'], dims['P'], dims['c']))
        nn.init.kaiming_uniform_(self.W_op, a=math.sqrt(5))

        self.layer_norm = nn.LayerNorm([dims['c']])

    def forward(self, prompt, img):
        if len(prompt.shape) == 1:
            prompt = prompt.view(-1, 1)
        prompt_encoded = self.prompt_embedding(prompt)  # [B x N_prompt x d]
        prompt_encoded = torch.sum(prompt_encoded, dim=1)  # [B x d]
        prompts_ops = torch.einsum('bd,dpc->bpc', prompt_encoded, self.W_op)  # [B x P x c]

        img_features = self.img_perceptor(img)  # [B, P]

        pred_embeded = torch.einsum('bp,bpc->bc', img_features, prompts_ops) # [B, c]

        pred_embeded = self.layer_norm(pred_embeded)

        t_embeddings = self.target_embedding.weight
        logits = pred_embeded @ t_embeddings.T
        return logits


from torch.nn import TransformerEncoder, TransformerEncoderLayer, TransformerDecoderLayer, TransformerDecoder


class VQAModelV11(nn.Module):

    @staticmethod
    def build(vocab,
              img_perceptor,
              d_model=16,
              encoder_ffn_dim=32,
              encoder_layers=2,
              encoder_attention_heads=2,
              encoder_dropout=0,
              decoder_ffn_dim=32,
              decoder_layers=2,
              decoder_attention_heads=2,
              decoder_dropout=0):
        encoder_layer = TransformerEncoderLayer(d_model, encoder_attention_heads, encoder_ffn_dim, encoder_dropout)
        transformer_encoder = TransformerEncoder(encoder_layer, encoder_layers)

        decoder_layer = TransformerDecoderLayer(d_model, decoder_attention_heads, decoder_ffn_dim, decoder_dropout)
        transformer_decoder = TransformerDecoder(decoder_layer, decoder_layers)
        return VQAModelV11(transformer_encoder, img_perceptor, transformer_decoder, vocab, d_model)

    def __init__(self, transformer_encoder, img_perceptor, transformer_decoder, vocab, d_model, pos_dropout=.1):
        super(VQAModelV11, self).__init__()
        # self.model_type = 'Transformer'
        # self.src_mask = None
        #
        self.img_perceptor = img_perceptor
        self.vocab = vocab
        V = len(vocab)
        tokens_embedding = Embedding(V, d_model, vocab.pad())
        self.tokens_embedding = tokens_embedding

        self.pos_encoder = PositionalEncoding(d_model, pos_dropout)
        self.transformer_encoder = transformer_encoder
        self.transformer_decoder = transformer_decoder

        self.d_model = d_model

    def forward(self, prompt, img, prev_tokens=None):
        prompt_embed = self.tokens_embedding(prompt) * math.sqrt(self.d_model)
        prompt_embed = self.pos_encoder(prompt_embed)
        x = prompt_embed

        img_embed = None
        if img is not None and self.img_perceptor is not None:
            img_embed = self.img_perceptor(img)
            comodal_embed = torch.cat([x, img_embed], dim=1)
            x = comodal_embed

        # compute padding mask
        encoder_padding_mask = prompt.eq(self.vocab.pad_index)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None
        elif img_embed is not None:
            B, F_img, d = img_embed.shape
            img_pad_mask = torch.zeros(B, F_img).to(device) == 1
            comodal_pad_mask = torch.cat([encoder_padding_mask, img_pad_mask], dim=1)
            encoder_padding_mask = comodal_pad_mask

        x = x.transpose(0, 1)
        encoder_output = self.transformer_encoder(x, src_key_padding_mask=encoder_padding_mask)
        # encoder_output shape = [N_in, B, d]

        B, N_in = prompt.shape
        bos_tensor = torch.ones(B, 1, dtype=torch.int64).to(device) * self.vocab.bos_index
        bos_tensor = self.tokens_embedding(bos_tensor).transpose(0, 1)
        decoder_output = self.transformer_decoder(bos_tensor, encoder_output, memory_key_padding_mask=encoder_padding_mask)
        decoder_output = decoder_output.transpose(0, 1)

        logits = F.linear(decoder_output, self.tokens_embedding.weight)
        return logits


class PositionalEncoding(nn.Module):

    def __init__(self, d_model, dropout=0.1, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


def build_embeddings(d, dataset, c=None):
    if dataset.prompt_mode == 'natural' and dataset.target_mode == 'natural':
        prompt_embeddings = Embedding(len(dataset.vocab), d, padding_idx=dataset.vocab.pad_index)
        target_embeddings = prompt_embeddings
    else:

        if dataset.prompt_mode == 'natural':
            V = len(dataset.vocab)
            prompt_pad_idx = dataset.vocab.pad_index
        else:
            V = len(dataset.concept_to_idx)
            prompt_pad_idx = None

        if dataset.target_mode == 'natural':
            L = len(dataset.vocab)
            target_pad_idx = dataset.vocab.pad_index
        else:
            L = len(dataset.cls_to_idx)
            target_pad_idx = None

        if c is None:
            c = d

        prompt_embeddings = Embedding(V, d, padding_idx=prompt_pad_idx)
        target_embeddings = Embedding(L, c, padding_idx=target_pad_idx)
    return prompt_embeddings, target_embeddings


def Embedding(num_embeddings, embedding_dim, padding_idx=None):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    # nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
    if padding_idx is not None:
        nn.init.constant_(m.weight[padding_idx], 0)
    return m

