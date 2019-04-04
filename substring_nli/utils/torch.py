import logging
import subprocess

import torch
from torch import nn
from torch.autograd import Variable

import numpy as np

from .. import config

logger = logging.getLogger(__name__)


def to_var(tensor, use_cuda, requires_grad=True):
    """Transform tensor into variable and transfer to GPU depending on flag"""
    if requires_grad:
        tensor.requires_grad_()

    if use_cuda:
        tensor = tensor.cuda()
    return tensor


def pack_forward(module, emb_batch, lengths, use_cuda=True, batch_first=True):
    """Based on: https://github.com/facebookresearch/InferSent/blob/4b7f9ec7192fc0eed02bc890a56612efc1fb1147/models.py

       Automatically sort and pck a padded sequence, feed it to an RNN and then
       unpack, re-pad and unsort it.

        Args:
            module: an instance of a torch RNN module
            batch: a pytorch tensor of dimension (batch_size, seq_len, emb_dim)
            lengths: a pytorch tensor of dimension (batch_size)"""

    sent = emb_batch
    sent_len = lengths.cpu().numpy()

    # Sort by length (keep idx)
    sent_len, idx_sort = np.sort(sent_len)[::-1], np.argsort(-sent_len)
    idx_unsort = np.argsort(idx_sort)

    idx_sort = (
        torch.from_numpy(idx_sort).cuda()
        if use_cuda
        else torch.from_numpy(idx_sort)
    )
    sent = sent.index_select(0, idx_sort)

    sent_len_list = sent_len.tolist()
    sent_len_list = [int(elem) for elem in sent_len_list]

    # Handling padding in Recurrent Networks
    sent_packed = nn.utils.rnn.pack_padded_sequence(
        sent, sent_len_list, batch_first=batch_first
    )

    sent_output, _ = module(sent_packed)

    sent_output = nn.utils.rnn.pad_packed_sequence(
        sent_output, batch_first=batch_first, padding_value=config.PAD_ID
    )[0]

    # Un-sort by length
    idx_unsort = (
        torch.from_numpy(idx_unsort).cuda()
        if use_cuda
        else torch.from_numpy(idx_unsort)
    )

    select_dim = 0 if batch_first else 1
    sent_output = sent_output.index_select(select_dim, idx_unsort)

    return sent_output


def get_norm(torch_tensor, dim=0, order=2):
    """Get the norm of a 2D tensor of the specified dim

    Parameters
    ----------
    torch_tensor : torch.tensor
        2-dimensional torch tensor
    dim : int, optional
    order : float, optional

    Returns
    -------
    torch_tensor : a 1-dimensional torch tensor containing the order-norms of
                   the given dimension

    """
    if order == 1:
        powered = torch.abs(torch_tensor)
    else:
        powered = torch.pow(torch_tensor, order)

    summed = torch.sum(powered, dim, keepdim=True)
    norm = torch.pow(summed, 1 / order)
    return norm


def normalize_embeddings(embeddings, dim=0, order=2) -> torch.nn.Embedding:
    """Normalize embeddings in the specified dimension

    Parameters
    ----------
    embeddings : torch.nn.Embedding
    dim : int, optional
    order : float, optional

    Returns
    -------
    torch.nn.Parameter(torch.nn.Embedding) : the normalized embedding object

    """
    # norm = get_norm(embeddings.weight, dim=dim, order=order)
    norm = torch.norm(embeddings.weight, p=order, dim=dim, keepdim=True)
    normalized = torch.div(embeddings.weight, norm)
    embeddings.weight = torch.nn.Parameter(normalized)

    return embeddings


def get_gpu_memory_map():
    """Get the current gpu usage.
    From https://discuss.pytorch.org/t/access-gpu-memory-usage-in-pytorch/3192/4

    Returns
    -------
    usage: dict
        Keys are device ids as integers.
        Values are memory usage as integers in MB.
    """
    result = subprocess.check_output(
        [
            "nvidia-smi",
            "--query-gpu=memory.used",
            "--format=csv,nounits,noheader",
        ],
        encoding="utf-8",
    )
    # Convert lines into a dictionary
    gpu_memory = [int(x) for x in result.strip().split("\n")]
    gpu_memory_map = dict(zip(range(len(gpu_memory)), gpu_memory))
    return gpu_memory_map


def get_free_gpu_index(max_memory=10, unallowed_gpus=None):
    """Get a random GPU with at most max_memory used"""

    if unallowed_gpus is None:
        unallowed_gpus = []

    gpu_memory_map = get_gpu_memory_map()
    for gpu_idx, memory_used in gpu_memory_map.items():
        if memory_used <= max_memory and gpu_idx not in unallowed_gpus:
            logger.debug(f"Using GPU {gpu_idx}")
            return gpu_idx
    logger.debug("No allowed free GPUs")
    return None
