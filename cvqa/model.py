from fairseq.models import transformer
from fairseq.models.transformer import TransformerModel, TransformerEncoder, TransformerDecoder

import argparse

from cvqa import fairseq_misc


def build_model(vocab, params):
    args = fariseq_transformer_args(params)
    tokens_embedding = fairseq_misc.build_embedding(vocab, params['d'])
    encoder = TransformerEncoder(args, vocab, tokens_embedding)
    decoder = TransformerDecoder(args, vocab, tokens_embedding,
                                 no_encoder_attn=getattr(args, 'no_cross_attention', False))
    return TransformerModel(args, encoder, decoder)


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

