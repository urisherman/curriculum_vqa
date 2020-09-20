import torch
import torchvision as tv
import math

from torch import nn

from fairseq.models import transformer
from fairseq.models.transformer import TransformerModel, TransformerEncoder, TransformerDecoder, EncoderOut

import argparse

from cvqa import fairseq_misc, utils

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class VQAModelV0(nn.Module):

    @staticmethod
    def build(vocab, params):
        args = fariseq_transformer_args(params)
        tokens_embedding = fairseq_misc.build_embedding(vocab, params['d'])
        encoder = TransformerEncoder(args, vocab, tokens_embedding)
        decoder = TransformerDecoder(args, vocab, tokens_embedding,
                                     no_encoder_attn=getattr(args, 'no_cross_attention', False))

        img_percept = Resnet18PerceptionModel(params['d'])
        return VQAModelV0(encoder, img_percept, decoder)

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
            encoder_out=comodal_seq,  # N_in x B x d
            encoder_padding_mask=comodal_pad_mask,  # B x N_in
            encoder_embedding=None,  # B x N_in x d
            encoder_states=[],  # List[N_in x B x d]
        )

        decoder_out = self.decoder(prev_output_tokens, encoder_out=decoder_in)
        #     decoder_out: (
        #         0: Tensor[B, No, V], --> real output, aka logits, the unnormalized scores over each token
        #         1: {
        #             attn: Tensor: [B, No, Ni],  --> cross attention
        #             inner_states: list[Tensor: [?, ?, d]]  --> looks like decoder internal layers
        #         }
        #     )
        return decoder_out[0]


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



class BasicImgModel(nn.Module):

    def __init__(self, output_dim, output_channels=None):
        super().__init__()

        self.backbone = tv.models.resnet18(pretrained=True)
        backbone_output = self.__backbone_forward(torch.rand(1, 3, 224, 224))
        B, C, H, W = backbone_output.shape

        if output_channels is not None:
            self.stem_conv = nn.Conv2d(C, output_channels, kernel_size=1, padding=0, bias=False)
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

        # pred_embeded = F.linear(img_features, prompt_op.T)  # [B, c]
        pred_embeded = torch.einsum('bp,bpc->bc', img_features, prompts_ops) # [B, c]

        pred_embeded = self.layer_norm(pred_embeded)

        t_embeddings = self.target_embedding.weight
        logits = pred_embeded @ t_embeddings.T
        return logits


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
    if padding_idx is not None:
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


def build_fairseq_encoder(vocab, tokens_embeddings, d=16, num_layers=2, ffn_dim=32, heads=2, layerdrop=0, args=None):
    if args is None:
        args = fariseq_transformer_args({'d': d})
    args.encoder_layers = num_layers
    args.encoder_attention_heads = heads
    args.encoder_ffn_embed_dim = ffn_dim
    args.encoder_layerdrop = layerdrop
    fairseq_encoder = TransformerEncoder(args, vocab, tokens_embeddings)
    return VQAFairseqEncoder(fairseq_encoder)


def build_fairseq_decoder(vocab, tokens_embeddings, d=16, num_layers=2, ffn_dim=32, heads=2, layerdrop=0, args=None):
    if args is None:
        args = fariseq_transformer_args({'d': d})
    args.decoder_layers = num_layers
    args.decoder_attention_heads = heads
    args.decoder_ffn_embed_dim = ffn_dim
    args.decoder_layerdrop = layerdrop
    fairseq_decoder = TransformerDecoder(args, vocab, tokens_embeddings)
    # return fairseq_decoder
    return VQAFairseqDecoder(fairseq_decoder)


class VQAFairseqDecoder(nn.Module):

    def __init__(self, base_decoder):
        super().__init__()
        self.base_decoder = base_decoder

    def forward(self, prompt_embedding, prompt_pad_mask, img_embedding, prev_output_tokens):
        """
        :param prompt_embedding: Tensor[B, N_prompt, d]
        :param img_embedding: Tensor[B, F_img, d]
        :param prev_output_tokens: Tensor[B, N_target, d]
        :return:
        """

        # concatenate encoded text and img
        comodal_embedding = torch.cat([prompt_embedding, img_embedding], dim=1)

        # concatenate pad mask
        B, F_img, _ = img_embedding.shape

        if prompt_pad_mask is None:
            comodal_pad_mask = None
        else:
            # TODO: Verify pads in the middle are not problematic
            img_pad_mask = utils.torch_zeros(B, F_img) == 1
            comodal_pad_mask = torch.cat([prompt_pad_mask, img_pad_mask], dim=1)

        decoder_in = EncoderOut(
            encoder_out=comodal_embedding.transpose(0, 1),  # N_in x B x d
            encoder_padding_mask=comodal_pad_mask,  # B x N_in
            encoder_embedding=None,  # B x N_in x d
            encoder_states=[],  # List[N_in x B x d]
        )

        decoder_out = self.base_decoder(prev_output_tokens, encoder_out=decoder_in)
        #     decoder_out: (
        #         0: Tensor[B, No, V], --> real output, aka logits, the unnormalized scores over each token
        #         1: {
        #             attn: Tensor: [B, No, Ni],  --> cross attention
        #             inner_states: list[Tensor: [?, ?, d]]  --> looks like decoder internal layers
        #         }
        #     )
        return decoder_out[0]


class VQAFairseqEncoder(nn.Module):

    def __init__(self, base_encoder):
        super().__init__()
        self.base_encoder = base_encoder

    def forward(self, prompt_tokens):
        """
        :param prompt_tokens: Tensor[B, N_prompt]
        :return:
        """
        encoder_out = self.base_encoder(prompt_tokens, src_lengths=None)
        prompt_embedding = encoder_out.encoder_out.transpose(0, 1)
        pad_mask = encoder_out.encoder_padding_mask

        return prompt_embedding, pad_mask


class VQAModelV1(nn.Module):

    @staticmethod
    def build(vocab, d=16, img_output_features=20):
        tokens_embeddings = fairseq_misc.build_embedding(vocab, d)
        prompt_encoder = build_fairseq_encoder(vocab, tokens_embeddings, d=d)
        vqa_decoder = build_fairseq_decoder(vocab, tokens_embeddings, d=d)
        img_perceptor = BasicImgModel(d, img_output_features)
        return VQAModelV1(prompt_encoder, img_perceptor, vqa_decoder, bos=vocab.bos_index, eos=vocab.eos_index)

    def __init__(self, prompt_encoder, img_pereptor, vqa_decoder, bos=0, eos=2):
        super().__init__()
        self.prompt_encoder = prompt_encoder
        self.img_pereptor = img_pereptor
        self.vqa_decoder = vqa_decoder
        self.bos = bos
        self.eos = eos

    def forward(self, prompt_tokens, img, prev_output_tokens=None):
        prompt_embedding, prompt_pad_mask = self.prompt_encoder(prompt_tokens)
        img_embedding = self.img_pereptor(img)

        B, N_in = prompt_tokens.shape
        bos_tensor = torch.ones(B, 1, dtype=torch.int64).to(device) * self.bos
        return self.vqa_decoder(prompt_embedding, prompt_pad_mask, img_embedding, bos_tensor)

    def forward_predict(self, prompt_tokens, img, target_limit=5):
        return self.forward(prompt_tokens, img)
        # prompt_embedding, prompt_pad_mask = self.prompt_encoder(prompt_tokens)
        # img_embedding = self.img_pereptor(img)
        #
        # bos_tensor = torch.tensor([self.bos]).to(device)
        # y_pred = torch.tensor([]).to(device)
        # for i in range(target_limit):
        #     prev_output_tokens = torch.cat([bos_tensor, y_pred])
        #     y_pred = self.vqa_decoder(prompt_embedding, prompt_pad_mask, img_embedding, prev_output_tokens)
        #     if y_pred[-1] == self.eos:
        #         break
        #
        # return y_pred