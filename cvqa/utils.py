import torch
import fairseq

from tqdm import tqdm

IS_CUDA = torch.cuda.is_available()

if IS_CUDA:
    device = torch.device("cuda")
else:
    device = torch.device("cpu")


def sample_to_cuda(sample):
    if IS_CUDA:
        return fairseq.utils.move_to_cuda(sample)
    else:
        return sample


def torch_zeros(*size, **kwargs):
    t = torch.zeros(*size, **kwargs)
    if IS_CUDA:
        t = t.cuda()
    return t


def to_device(t):
    if IS_CUDA:
        t = t.cuda()
    return t