import torch
from fairseq.models.transformer import Embedding
from fairseq import utils


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




def label_smoothed_nll_loss(lprobs, target, epsilon, ignore_index=None, reduce=True):
    """
    Copied from fairseq.criterion.label_smoothed_cross_entropy
    :param lprobs:
    :param target:
    :param epsilon:
    :param ignore_index:
    :param reduce:
    :return:
    """
    if target.dim() == lprobs.dim() - 1:
        target = target.unsqueeze(-1)
    nll_loss = -lprobs.gather(dim=-1, index=target)
    smooth_loss = -lprobs.sum(dim=-1, keepdim=True)
    if ignore_index is not None:
        non_pad_mask = target.ne(ignore_index)
        nll_loss = nll_loss[non_pad_mask]
        smooth_loss = smooth_loss[non_pad_mask]
    else:
        nll_loss = nll_loss.squeeze(-1)
        smooth_loss = smooth_loss.squeeze(-1)
    if reduce:
        nll_loss = nll_loss.sum()
        smooth_loss = smooth_loss.sum()
    eps_i = epsilon / lprobs.size(-1)
    loss = (1. - epsilon) * nll_loss + eps_i * smooth_loss
    return loss, nll_loss


def clear_cuda(args, num_updates):
    """
    Copied from fairseq.trainer.train_step()

    Clear CUDA cache to reduce memory fragmentation
    :param args:
    :param num_updates:
    :return:
    """
    if (
        args.empty_cache_freq > 0
        and (
            (num_updates + args.empty_cache_freq - 1)
            % args.empty_cache_freq
        )
        == 0
        and torch.cuda.is_available()
        and not args.cpu
    ):
        torch.cuda.empty_cache()
