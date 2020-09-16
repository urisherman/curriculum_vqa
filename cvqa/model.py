import torch
import torchvision as tv
import torch.nn.functional as F
import math

from torch import nn

from fairseq.models import transformer
from fairseq.models.transformer import TransformerModel, TransformerEncoder, TransformerDecoder, EncoderOut

import argparse

from cvqa import fairseq_misc

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def build_model(vocab, params):
    args = fariseq_transformer_args(params)
    tokens_embedding = fairseq_misc.build_embedding(vocab, params['d'])
    encoder = TransformerEncoder(args, vocab, tokens_embedding)
    decoder = TransformerDecoder(args, vocab, tokens_embedding,
                                 no_encoder_attn=getattr(args, 'no_cross_attention', False))

    # return TransformerModel(args, encoder, decoder)
    img_percept = Resnet18PerceptionModel(params['d'])
    return VQAModelV0(encoder, img_percept, decoder)


class VQAModelV0(nn.Module):

    def __init__(self, text_encoder, img_pereptor, decoder):
        super().__init__()
        self.text_encoder = text_encoder
        self.img_pereptor = img_pereptor
        self.decoder = decoder

    def forward(self, src_tokens, src_img, prev_output_tokens):
        encoder_out = self.text_encoder(src_tokens, src_lengths=None)
        perception_out = self.img_pereptor(src_img)
        B, Z, d = perception_out.shape

        # concatenate encoded text and img
        comodal_seq = torch.cat([encoder_out.encoder_out, torch.transpose(perception_out, 0, 1)], dim=0)

        # concatenate pad mask
        img_pad_mask = torch.zeros(B, Z).to(device) == 1
        comodal_pad_mask = torch.cat([encoder_out.encoder_padding_mask, img_pad_mask], dim=1)

        decoder_in = EncoderOut(
            encoder_out=comodal_seq,  # T x B x C
            encoder_padding_mask=comodal_pad_mask,  # B x T
            encoder_embedding=None,  # B x T x C
            encoder_states=[],  # List[T x B x C]
        )

        decoder_out = self.decoder(prev_output_tokens, encoder_out=decoder_in)
        return decoder_out


class Resnet18PerceptionModel(nn.Module):

    def __init__(self, output_dim):
        super().__init__()
        self.backbone = tv.models.resnet18(pretrained=True)
        self.output_conv = nn.Conv2d(256, 30, kernel_size=3, padding=3, bias=False)

        output_conv_shape = self.output_conv(torch.rand(1, 256, 14, 14)).data.shape

        self.fc = nn.Linear(output_conv_shape[2] * output_conv_shape[3], output_dim)

    def forward(self, x):
        bb = self.backbone

        x = bb.conv1(x)
        x = bb.bn1(x)
        x = bb.relu(x)
        x = bb.maxpool(x)

        x = bb.layer1(x)
        x = bb.layer2(x)
        x = bb.layer3(x)
        #         x = bb.layer4(x)

        #         x = bb.avgpool(x)
        #         x = torch.flatten(x, 1)
        #         x = bb.fc(x)

        x = self.output_conv(x)
        return self.fc(torch.flatten(x, start_dim=-2))


class VQAConcept2ClassModel(nn.Module):

    def __init__(self, num_concepts, num_classes, img_perceptor=None):
        super().__init__()
        self.num_concepts = num_concepts
        self.num_classes = num_classes

        if img_perceptor is None:
            img_perceptor = BasicImgModel(20)
        self.img_perceptor = img_perceptor

        self.prompt_reader_layer = nn.Linear(num_concepts, num_classes)

        img_embedding = img_perceptor(torch.rand(1, 3, 224, 224))
        B, d = img_embedding.shape
        self.fc = nn.Linear(d, num_classes)

    def read(self, prompt):
        concept = prompt
        one_hot_concept = torch.nn.functional.one_hot(concept, num_classes=self.num_concepts).to(device).float()
        return one_hot_concept

    def answer(self, prompt_embedding, img_embedding):
        mask = self.prompt_reader_layer(prompt_embedding)
        output = self.fc(torch.flatten(img_embedding, start_dim=1))
        return output * mask

    def forward(self, prompt, img):
        prompt_embedding = self.read(prompt)
        img_embedding = self.img_perceptor(img)
        return self.answer(prompt_embedding, img_embedding)


class BasicImgModel(nn.Module):

    def __init__(self, output_dim):
        super().__init__()
        self.backbone = tv.models.resnet18(pretrained=True)
        backbone_output = self.__backbone_forward(torch.rand(1, 3, 224, 224))
        B, C, H, W = backbone_output.shape
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
        img_features = self.__backbone_forward(img)
        output = self.fc(torch.flatten(img_features, start_dim=1))
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

        from torch.nn import TransformerEncoder, TransformerEncoderLayer
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
        self.W_op = nn.Parameter(torch.Tensor(dims['P'], dims['d'], dims['c']))
        nn.init.kaiming_uniform_(self.W_op, a=math.sqrt(5))

        self.layer_norm = nn.LayerNorm([dims['c']])

    def forward(self, prompt, img):
        if len(prompt.shape) == 1:
            prompt = prompt.view(-1, 1)
        prompt_encoded = self.prompt_embedding(prompt)  # [B x N_prompt x d]
        prompt_encoded = torch.sum(prompt_encoded, dim=1)  # [B x d]
        prompt_op = torch.einsum('pdc,bd->pc', self.W_op, prompt_encoded)  # [P x c]

        img_features = self.img_perceptor(img)  # [B, P]

        pred_embeded = F.linear(img_features, prompt_op.T)  # [B, c]
        pred_embeded = self.layer_norm(pred_embeded)

        t_embeddings = self.target_embedding.weight
        logits = pred_embeded @ t_embeddings.T
        return logits


def build_embeddings(d, dataset, c=None):
    if dataset.prompt_mode == 'natural' and dataset.target_mode == 'natural':
        prompt_embeddings = Embedding(len(dataset.vocab), d, padding_idx=0)
        target_embeddings = prompt_embeddings
    else:
        if dataset.prompt_mode == 'natural':
            V = len(dataset.vocab)
        else:
            V = len(dataset.concept_to_idx)

        if dataset.target_mode == 'natural':
            L = len(dataset.vocab)
        else:
            L = len(dataset.cls_to_idx)

        if c is None:
            c = d

        prompt_embeddings = Embedding(V, d, padding_idx=0)
        target_embeddings = Embedding(L, c, padding_idx=0)

    return prompt_embeddings, target_embeddings


def Embedding(num_embeddings, embedding_dim, padding_idx):
    m = nn.Embedding(num_embeddings, embedding_dim, padding_idx=padding_idx)
    nn.init.normal_(m.weight, mean=0, std=embedding_dim ** -0.5)
    # nn.init.kaiming_uniform_(m.weight, a=math.sqrt(5))
    nn.init.constant_(m.weight[padding_idx], 0)
    return m


def fariseq_transformer_args(params):
    # parser = options.get_training_parser()
    parser = argparse.ArgumentParser()
    # TransformerModel.add_args(parser)

    args = parser.parse_args([])
    transformer.base_architecture(args)

    args.seed = 1

    args.label_smoothing = 0.1
    args.max_source_positions = transformer.DEFAULT_MAX_SOURCE_POSITIONS
    args.max_target_positions = transformer.DEFAULT_MAX_TARGET_POSITIONS

    args.dropout = 0.3

    args.encoder_embed_dim = params['d']
    args.decoder_embed_dim = params['d']

    args.encoder_layers = 2
    args.encoder_layerdrop = 0
    args.encoder_attention_heads = 4

    args.decoder_input_dim = params['d']
    args.decoder_layers = 2
    args.decoder_layerdrop = 0
    args.decoder_attention_heads = 4
    args.share_decoder_input_output_embed = True
    args.decoder_output_dim = params['d']
    return args
