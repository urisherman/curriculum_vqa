from argparse import Namespace

import torch
from collections import namedtuple
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from fairseq import utils
from fairseq.models import (
    FairseqEncoder
)
from fairseq.models.transformer import TransformerModel, Embedding, EncoderOut, TransformerDecoder
from fairseq.modules import (
    LayerNorm,
    PositionalEmbedding,
    SinusoidalPositionalEmbedding,
    TransformerEncoderLayer
)
import random

FAIRSEQ_DEFAULT_ARGS = {'no_progress_bar': False, 'log_interval': 1000, 'log_format': None, 'tensorboard_logdir': '', 'seed': 1, 'cpu': False, 'fp16': False, 'memory_efficient_fp16': False, 'fp16_init_scale': 128, 'fp16_scale_window': None, 'fp16_scale_tolerance': 0.0, 'min_loss_scale': 0.0001, 'threshold_loss_scale': None, 'user_dir': None, 'empty_cache_freq': 0, 'criterion': 'label_smoothed_cross_entropy', 'tokenizer': None, 'bpe': None, 'optimizer': 'adam', 'lr_scheduler': 'inverse_sqrt', 'task': 'translation', 'num_workers': 1, 'skip_invalid_size_inputs_valid_test': False, 'max_tokens': 4096, 'max_sentences': None, 'required_batch_size_multiple': 8, 'dataset_impl': None, 'train_subset': 'train', 'valid_subset': 'valid', 'validate_interval': 1, 'fixed_validation_seed': None, 'disable_validation': False, 'max_tokens_valid': 4096, 'max_sentences_valid': None, 'curriculum': 0, 'distributed_world_size': 1, 'distributed_rank': 0, 'distributed_backend': 'nccl', 'distributed_init_method': None, 'distributed_port': -1, 'device_id': 0, 'distributed_no_spawn': False, 'ddp_backend': 'c10d', 'bucket_cap_mb': 25, 'fix_batches_to_gpus': False, 'find_unused_parameters': False, 'fast_stat_sync': False, 'arch': 'transformer_iwslt_de_en', 'max_epoch': 0, 'max_update': 0, 'clip_norm': 0.0, 'sentence_avg': False, 'update_freq': [1], 'lr': [0.0005], 'min_lr': -1, 'use_bmuf': False, 'save_dir': 'checkpoints', 'restore_file': 'checkpoint_last.pt', 'reset_dataloader': False, 'reset_lr_scheduler': False, 'reset_meters': False, 'reset_optimizer': False, 'optimizer_overrides': '{}', 'save_interval': 1, 'save_interval_updates': 0, 'keep_interval_updates': -1, 'keep_last_epochs': -1, 'no_save': False, 'no_epoch_checkpoints': False, 'no_last_checkpoints': False, 'no_save_optimizer_state': False, 'best_checkpoint_metric': 'loss', 'maximize_best_checkpoint_metric': False, 'no_token_positional_embeddings': False, 'no_cross_attention': False, 'cross_self_attention': False, 'layer_wise_attention': False, 'encoder_layerdrop': 0, 'decoder_layerdrop': 0, 'encoder_layers_to_keep': None, 'decoder_layers_to_keep': None, 'label_smoothing': 0.1, 'adam_betas': '(0.9, 0.999)', 'adam_eps': 1e-08, 'weight_decay': 0.0001, 'warmup_updates': 4000, 'warmup_init_lr': -1, 'data': '/Users/urisherman/Work/workspace/curriculum_vqa/data-bin/iwslt14.tokenized.de-en', 'source_lang': 'de', 'target_lang': 'en', 'lazy_load': False, 'raw_text': False, 'load_alignments': False, 'left_pad_source': True, 'left_pad_target': False, 'max_source_positions': 1024, 'max_target_positions': 1024, 'upsample_primary': 1, 'truncate_source': False, 'share_decoder_input_output_embed': True, 'dropout': 0.3, 'encoder_embed_dim': 512, 'encoder_ffn_embed_dim': 1024, 'encoder_attention_heads': 4, 'encoder_layers': 6, 'decoder_embed_dim': 512, 'decoder_ffn_embed_dim': 1024, 'decoder_attention_heads': 4, 'decoder_layers': 6, 'encoder_embed_path': None, 'encoder_normalize_before': False, 'encoder_learned_pos': False, 'decoder_embed_path': None, 'decoder_normalize_before': False, 'decoder_learned_pos': False, 'attention_dropout': 0.0, 'activation_dropout': 0.0, 'activation_fn': 'relu', 'adaptive_softmax_cutoff': None, 'adaptive_softmax_dropout': 0, 'share_all_embeddings': False, 'adaptive_input': False, 'decoder_output_dim': 512, 'decoder_input_dim': 512, 'no_scale_embedding': False, 'layernorm_embedding': False}


def build_transformer(
        vocab,
        d=16,
        ffn_dim=32,
        encoder_layers=2,
        decoder_layers=2,
        attention_heads=2):

    FakeTask = namedtuple('FakeTask', 'source_dictionary target_dictionary')

    fake_task = FakeTask(vocab, vocab)
    args = Namespace(**FAIRSEQ_DEFAULT_ARGS)

    args.share_all_embeddings = True

    args.encoder_embed_dim = d
    args.encoder_ffn_embed_dim = ffn_dim
    args.encoder_layers = encoder_layers
    args.encoder_attention_heads = attention_heads

    args.decoder_embed_dim = d
    args.decoder_ffn_embed_dim = ffn_dim
    args.decoder_attention_heads = attention_heads
    args.decoder_layers = decoder_layers

    args.tie_adaptive_weights = False
    args.decoder_output_dim = d
    args.decoder_input_dim = d

    return TransformerModel.build_model(args, fake_task)


def build_decoder(
        vocab,
        tokens_embeddings,
        ffn_dim=32,
        layers=2,
        attention_heads=2):

    args = Namespace(**FAIRSEQ_DEFAULT_ARGS)

    d = tokens_embeddings.embedding_dim
    args.share_all_embeddings = True

    args.encoder_embed_dim = d

    args.decoder_embed_dim = d
    args.decoder_ffn_embed_dim = ffn_dim
    args.decoder_attention_heads = attention_heads
    args.decoder_layers = layers

    args.tie_adaptive_weights = False
    args.decoder_output_dim = d
    args.decoder_input_dim = d

    return TransformerDecoder(args, vocab, tokens_embeddings)


def build_vqa_encoder(
        vocab,
        tokens_embeddings,
        ffn_dim=32,
        layers=2,
        attention_heads=2):

    args = Namespace(**FAIRSEQ_DEFAULT_ARGS)

    d = tokens_embeddings.embedding_dim

    args.share_all_embeddings = True

    args.encoder_embed_dim = d
    args.encoder_ffn_embed_dim = ffn_dim
    args.encoder_layers = layers
    args.encoder_attention_heads = attention_heads

    args.tie_adaptive_weights = False

    return VQATransformerEncoder(args, vocab, tokens_embeddings)


def build_embedding(dictionary, embed_dim, path=None):
    """
    Copied from fairseq.models.transformer
    :param dictionary:
    :param embed_dim:
    :param path:
    :return:
    """
    num_embeddings = len(dictionary)
    padding_idx = dictionary.pad()
    emb = Embedding(num_embeddings, embed_dim, padding_idx)
    # if provided, load from preloaded dictionaries
    if path:
        embed_dict = utils.parse_embedding(path)
        utils.load_embedding(embed_dict, dictionary, emb)
    return emb


if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


class VQATransformerEncoder(FairseqEncoder):
    """
    --- Copied and adjusted from fairseq ---

    Transformer encoder consisting of *args.encoder_layers* layers. Each layer
    is a :class:`TransformerEncoderLayer`.

    Args:
        args (argparse.Namespace): parsed command-line arguments
        dictionary (~fairseq.data.Dictionary): encoding dictionary
        embed_tokens (torch.nn.Embedding): input embedding
    """

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(dictionary)
        self.register_buffer('version', torch.Tensor([3]))

        self.dropout = args.dropout
        self.encoder_layerdrop = args.encoder_layerdrop

        embed_dim = embed_tokens.embedding_dim
        self.padding_idx = embed_tokens.padding_idx
        self.max_source_positions = args.max_source_positions

        self.embed_tokens = embed_tokens

        self.embed_scale = 1.0 if args.no_scale_embedding else math.sqrt(embed_dim)

        self.embed_positions = PositionalEmbedding(
            args.max_source_positions, embed_dim, self.padding_idx,
            learned=args.encoder_learned_pos,
        ) if not args.no_token_positional_embeddings else None

        self.layer_wise_attention = getattr(args, 'layer_wise_attention', False)

        self.layers = nn.ModuleList([])
        self.layers.extend([
            TransformerEncoderLayer(args)
            for i in range(args.encoder_layers)
        ])

        if args.encoder_normalize_before:
            self.layer_norm = LayerNorm(embed_dim)
        else:
            self.layer_norm = None
        if getattr(args, 'layernorm_embedding', False):
            self.layernorm_embedding = LayerNorm(embed_dim)
        else:
            self.layernorm_embedding = None

    def forward_embedding(self, src_tokens):
        # embed tokens and positions
        x = embed = self.embed_scale * self.embed_tokens(src_tokens)
        if self.embed_positions is not None:
            x = embed + self.embed_positions(src_tokens)
        if self.layernorm_embedding:
            x = self.layernorm_embedding(x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        return x, embed

    def forward(self, src_tokens, img_embedding, cls_input=None, return_all_hiddens=False, **unused):
        """
        Args:
            src_tokens (LongTensor): tokens in the source language of shape
                `(batch, src_len)`
            img_embedding (torch.FloatTensor):
                shape `(batch, F_img, d)`
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).

        Returns:
            namedtuple:
                - **encoder_out** (Tensor): the last encoder layer's output of
                  shape `(src_len, batch, embed_dim)`
                - **encoder_padding_mask** (ByteTensor): the positions of
                  padding elements of shape `(batch, src_len)`
                - **encoder_embedding** (Tensor): the (scaled) embedding lookup
                  of shape `(batch, src_len, embed_dim)`
                - **encoder_states** (List[Tensor]): all intermediate
                  hidden states of shape `(src_len, batch, embed_dim)`.
                  Only populated if *return_all_hiddens* is True.
        """
        if self.layer_wise_attention:
            return_all_hiddens = True

        x, encoder_embedding = self.forward_embedding(src_tokens)

        comodal_embedding = torch.cat([x, img_embedding], dim=1)
        x = comodal_embedding

        # B x T x C -> T x B x C
        x = x.transpose(0, 1)

        # compute padding mask
        encoder_padding_mask = src_tokens.eq(self.padding_idx)
        if not encoder_padding_mask.any():
            encoder_padding_mask = None
        else:
            B, F_img, d = img_embedding.shape
            img_pad_mask = torch.zeros(B, F_img).to(device) == 1
            comodal_pad_mask = torch.cat([encoder_padding_mask, img_pad_mask], dim=1)
            encoder_padding_mask = comodal_pad_mask

        encoder_states = [] if return_all_hiddens else None

        # encoder layers
        for layer in self.layers:
            # add LayerDrop (see https://arxiv.org/abs/1909.11556 for description)
            dropout_probability = random.uniform(0, 1)
            if not self.training or (dropout_probability > self.encoder_layerdrop):
                x = layer(x, encoder_padding_mask)
                if return_all_hiddens:
                    encoder_states.append(x)

        if self.layer_norm:
            x = self.layer_norm(x)
            if return_all_hiddens:
                encoder_states[-1] = x

        return EncoderOut(
            encoder_out=x,  # T x B x C
            encoder_padding_mask=encoder_padding_mask,  # B x T
            encoder_embedding=encoder_embedding,  # B x T x C
            encoder_states=encoder_states,  # List[T x B x C]
        )

    def reorder_encoder_out(self, encoder_out, new_order):
        """
        Reorder encoder output according to *new_order*.

        Args:
            encoder_out: output from the ``forward()`` method
            new_order (LongTensor): desired order

        Returns:
            *encoder_out* rearranged according to *new_order*
        """
        if encoder_out.encoder_out is not None:
            encoder_out = encoder_out._replace(
                encoder_out=encoder_out.encoder_out.index_select(1, new_order)
            )
        if encoder_out.encoder_padding_mask is not None:
            encoder_out = encoder_out._replace(
                encoder_padding_mask=encoder_out.encoder_padding_mask.index_select(0, new_order)
            )
        if encoder_out.encoder_embedding is not None:
            encoder_out = encoder_out._replace(
                encoder_embedding=encoder_out.encoder_embedding.index_select(0, new_order)
            )
        if encoder_out.encoder_states is not None:
            for idx, state in enumerate(encoder_out.encoder_states):
                encoder_out.encoder_states[idx] = state.index_select(1, new_order)
        return encoder_out

    def max_positions(self):
        """Maximum input length supported by the encoder."""
        if self.embed_positions is None:
            return self.max_source_positions
        return min(self.max_source_positions, self.embed_positions.max_positions())

    def buffered_future_mask(self, tensor):
        dim = tensor.size(0)
        if not hasattr(self, '_future_mask') or self._future_mask is None or self._future_mask.device != tensor.device:
            self._future_mask = torch.triu(utils.fill_with_neg_inf(tensor.new(dim, dim)), 1)
            if self._future_mask.size(0) < dim:
                self._future_mask = torch.triu(utils.fill_with_neg_inf(self._future_mask.resize_(dim, dim)), 1)
        return self._future_mask[:dim, :dim]

    def upgrade_state_dict_named(self, state_dict, name):
        """Upgrade a (possibly old) state dict for new versions of fairseq."""
        if isinstance(self.embed_positions, SinusoidalPositionalEmbedding):
            weights_key = '{}.embed_positions.weights'.format(name)
            if weights_key in state_dict:
                print('deleting {0}'.format(weights_key))
                del state_dict[weights_key]
            state_dict['{}.embed_positions._float_tensor'.format(name)] = torch.FloatTensor(1)
        for i in range(len(self.layers)):
            # update layer norms
            self.layers[i].upgrade_state_dict_named(state_dict, "{}.layers.{}".format(name, i))

        version_key = '{}.version'.format(name)
        if utils.item(state_dict.get(version_key, torch.Tensor([1]))[0]) < 2:
            # earlier checkpoints did not normalize after the stack of layers
            self.layer_norm = None
            self.normalize = False
            state_dict[version_key] = torch.Tensor([1])
        return state_dict