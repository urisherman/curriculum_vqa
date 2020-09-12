import torch
import torchvision as tv

from torch import nn

from fairseq.models import transformer
from fairseq.models.transformer import TransformerModel, TransformerEncoder, TransformerDecoder, EncoderOut

import argparse

from cvqa import fairseq_misc


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
        img_pad_mask = torch.zeros(B, Z) == 1
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
