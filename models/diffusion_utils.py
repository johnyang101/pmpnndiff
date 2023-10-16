import torch
import torch.nn.functional as F
import numpy as np
from inspect import isfunction
from tqdm import tqdm
from torchtyping import TensorType

"""
Based in part on: https://github.com/lucidrains/denoising-diffusion-pytorch/blob/5989f4c77eafcdc6be0fb4739f0f277a6dd7f7d8/denoising_diffusion_pytorch/denoising_diffusion_pytorch.py#L281
"""
eps = 1e-8

'''
General tutorial:

Start with _train_loss function, sample, and sample_chain.
'''


def perplexity(logits):
    log_prob = logits - torch.logsumexp(logits, dim=1, keepdim=True);
    return torch.exp(-(log_prob * log_prob.exp()).sum(dim=1))


def norm_probs(unnormed_probs):
    '''
    Takes in a [B, N, K] shape vector
    
    returns a normalized [B, N, K] shape vector
    
    '''
    norm_factor = unnormed_probs.sum([2], keepdim=True)
    normed_probs = unnormed_probs / norm_factor
    return normed_probs
    
def logits_to_onehot(dist):
    '''
    Takes in a [B, N, K] size torch.tensor
    
    Returns a [B, N, K] size torch.tensor with one hot vectors across Class dim (2).
    
    '''
    K = 20
    return torch.nn.functional.one_hot(torch.argmax(dist, dim=2), num_classes=K).float()


def sum_except_batch(x, num_dims=1):
    '''
    Sums all dimensions except the first.

    Args:
        x: Tensor, shape (batch_size, ...)
        num_dims: int, number of batch dims (default=1)

    Returns:
        x_sum: Tensor, shape (batch_size,)
    '''
    return x.reshape(*x.shape[:num_dims], -1).sum(-1)


def sum_except_batch_mask(x, mask, num_dims=1):
    '''
    Sums all dimensions except the first.

    Args:
        x: Tensor, shape (batch_size, ...)
        num_dims: int, number of batch dims (default=1)

    Returns:
        x_sum: Tensor, shape (batch_size,)
    '''
    assert x.size() == mask.size(), f'x size {x.size()} not equal to mask size {mask.size()}'
    masked = (x * mask)
    return masked.reshape(*x.shape[:num_dims], -1).sum(-1)

def log_1_min_a(a):
    return torch.log(1 - a.exp() + 1e-40)


def log_add_exp(a, b):
    maximum = torch.max(a, b)
    return maximum + torch.log(torch.exp(a - maximum) + torch.exp(b - maximum))


def exists(x):
    return x is not None


def extract(a: TensorType['A'], t: TensorType['T'], x_shape):
    b, *_ = t.shape
    out: TensorType['T'] = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def log_categorical(log_x_start, log_prob):
    
    return (log_x_start.exp() * log_prob).sum(dim=1)


def index_to_log_onehot(x: TensorType['B', 'N'], num_classes) -> TensorType['B', 'K', 'N']:
    assert x.max().item() < num_classes, \
        f'Error: {x.max().item()} >= {num_classes}'
    x_onehot = F.one_hot(x, num_classes)

    permute_order = (0, -1) + tuple(range(1, len(x.size())))

    x_onehot = x_onehot.permute(permute_order)

    log_x = torch.log(x_onehot.float().clamp(min=1e-30))

    return log_x


def log_onehot_to_index(log_x: TensorType['B', 'K', 'N']) -> TensorType['B', 'N']:
    return log_x.argmax(1)


def cosine_beta_schedule(timesteps, s = 0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = np.linspace(0, steps, steps)
    alphas_cumprod = np.cos(((x / steps) + s) / (1 + s) * np.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    alphas = (alphas_cumprod[1:] / alphas_cumprod[:-1])

    alphas = np.clip(alphas, a_min=0.001, a_max=1.)

    # Use sqrt of this, so the alpha in our paper is the alpha_sqrt from the
    # Gaussian diffusion in Ho et al.
    alphas = np.sqrt(alphas)
    return alphas

pi = 3.141592653589793
Tensor = torch.Tensor

def positional_embedding(N, embed_size):
    """positional_embedding creates sine / cosine positional embeddings as described
    in `Attention is all you need'
    
    Args:
        N: number of positions to embed
        embed_size: dimension of the embeddings to create
                    NOTE: USE EVEN NUMBERS SINCE FLOOR DIVISION USED.
    
    Returns:
        positional embedding of shape [N, embed_size]
    """
    idx = torch.arange(N)
    half_embed_size = torch.arange(embed_size//2)
    pos_embedding_sin = torch.sin(idx[:,None] * pi / (N**(2*half_embed_size[None]/embed_size)))
    pos_embedding_cos = torch.cos(idx[:,None] * pi / (N**(2*half_embed_size[None]/embed_size)))
    pos_embedding = torch.concat([
        pos_embedding_sin, pos_embedding_cos], axis=-1)
    return pos_embedding

'''
MSFT VQ-Diffusion Utils

'''

def alpha_schedule(time_step, N=100, att_1 = 0.99999, att_T = 0.000009, ctt_1 = 0.000009, ctt_T = 0.99999):
    att = np.arange(0, time_step)/(time_step-1)*(att_T - att_1) + att_1
    att = np.concatenate(([1], att))
    at = att[1:]/att[:-1]
    ctt = np.arange(0, time_step)/(time_step-1)*(ctt_T - ctt_1) + ctt_1
    ctt = np.concatenate(([0], ctt))
    one_minus_ctt = 1 - ctt
    one_minus_ct = one_minus_ctt[1:] / one_minus_ctt[:-1]
    ct = 1-one_minus_ct
    bt = (1-at-ct)/N
    att = np.concatenate((att[1:], [1]))
    ctt = np.concatenate((ctt[1:], [0]))
    btt = (1-att-ctt)/N
    return at, bt, ct, att, btt, ctt