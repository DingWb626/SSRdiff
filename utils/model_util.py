# --------------------------------------------------------
# References:
# Lightning-DiT: https://github.com/hustvl/LightningDiT
# --------------------------------------------------------

from math import pi

import torch
from torch import nn
import numpy as np

from einops import rearrange, repeat


def broadcat(tensors, dim = -1):
    num_tensors = len(tensors)
    shape_lens = set(list(map(lambda t: len(t.shape), tensors)))
    assert len(shape_lens) == 1, 'tensors must all have the same number of dimensions'
    shape_len = list(shape_lens)[0]
    dim = (dim + shape_len) if dim < 0 else dim
    dims = list(zip(*map(lambda t: list(t.shape), tensors)))
    expandable_dims = [(i, val) for i, val in enumerate(dims) if i != dim]
    assert all([*map(lambda t: len(set(t[1])) <= 2, expandable_dims)]), 'invalid dimensions for broadcastable concatentation'
    max_dims = list(map(lambda t: (t[0], max(t[1])), expandable_dims))
    expanded_dims = list(map(lambda t: (t[0], (t[1],) * num_tensors), max_dims))
    expanded_dims.insert(dim, (dim, dims[dim]))
    expandable_shapes = list(zip(*map(lambda t: t[1], expanded_dims)))
    tensors = list(map(lambda t: t[0].expand(*t[1]), zip(tensors, expandable_shapes)))
    return torch.cat(tensors, dim = dim)



class VisionRotaryEmbedding(nn.Module):
    def __init__(
        self,
        dim,
        pt_seq_len,
        ft_seq_len=None,
        custom_freqs = None,
        freqs_for = 'lang',
        theta = 10000,
        max_freq = 10,
        num_freqs = 1,
    ):
        super().__init__()
        if custom_freqs:
            freqs = custom_freqs
        elif freqs_for == 'lang':
            freqs = 1. / (theta ** (torch.arange(0, dim, 2)[:(dim // 2)].float() / dim))
        elif freqs_for == 'pixel':
            freqs = torch.linspace(1., max_freq / 2, dim // 2) * pi
        elif freqs_for == 'constant':
            freqs = torch.ones(num_freqs).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        if ft_seq_len is None: ft_seq_len = pt_seq_len
        t = torch.arange(ft_seq_len) / ft_seq_len * pt_seq_len

        freqs_h = torch.einsum('..., f -> ... f', t, freqs)
        freqs_h = repeat(freqs_h, '... n -> ... (n r)', r = 2)

        freqs_w = torch.einsum('..., f -> ... f', t, freqs)
        freqs_w = repeat(freqs_w, '... n -> ... (n r)', r = 2)

        freqs = broadcat((freqs_h[:, None, :], freqs_w[None, :, :]), dim = -1)

        self.register_buffer("freqs_cos", freqs.cos())
        self.register_buffer("freqs_sin", freqs.sin())

    def forward(self, t, start_index = 0):
        rot_dim = self.freqs_cos.shape[-1]
        end_index = start_index + rot_dim
        assert rot_dim <= t.shape[-1], f'feature dimension {t.shape[-1]} is not of sufficient size to rotate in all the positions {rot_dim}'
        t_left, t, t_right = t[..., :start_index], t[..., start_index:end_index], t[..., end_index:]
        t = (t * self.freqs_cos) + (rotate_half(t) * self.freqs_sin)
        return torch.cat((t_left, t, t_right), dim = -1)


# 如果你原来已经有 rotate_half 就不用再写
def rotate_half(x):
    x1 = x[..., ::2]
    x2 = x[..., 1::2]
    return torch.stack((-x2, x1), dim=-1).reshape_as(x)


class VisionRotaryEmbeddingFast(nn.Module):
    """
    时间序列版 RoPE：
    - 不再做 2D broadcat（没有 H×W 网格），只做 1D 序列；
    - 仍然支持 num_cls_token，在前面插入 cls 的“零角度”位置；
    - 不再写死 .cuda()，通过 register_buffer 让 model.to(device) 自动搬运。

    注意：这里的 dim 还是和你之前一样，传的是 `half_head_dim`，
    最终作用在 q/k 上的最后一维是 `head_dim = 2 * dim`。
    """

    def __init__(
            self,
            dim,  # 传 half_head_dim
            pt_seq_len=16,
            ft_seq_len=None,
            custom_freqs=None,
            freqs_for='lang',
            theta=10000,
            max_freq=10,
            num_freqs=1,  # 基本可以不用管
            num_cls_token=0
    ):
        super().__init__()

        # 最终真正作用在 q/k 上的维度
        head_dim = dim * 2  # 对应你的 head_dim

        # -------- 构建频率 freqs: 长度 = head_dim // 2 --------
        if custom_freqs is not None:
            freqs = custom_freqs.float()
        elif freqs_for == 'lang':
            # 标准 1D RoPE：用 head_dim 做分母
            freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim))
        elif freqs_for == 'pixel':
            # 理论上也能用在时间序列上，当作另一种频率分布
            freqs = torch.linspace(1., max_freq / 2, head_dim // 2) * pi
        elif freqs_for == 'constant':
            # 固定频率，全 1
            freqs = torch.ones(head_dim // 2).float()
        else:
            raise ValueError(f'unknown modality {freqs_for}')

        # -------- 序列长度映射（支持预训练/微调长度不同）--------
        if ft_seq_len is None:
            ft_seq_len = pt_seq_len

        # t: [L]，L 为当前使用的时间步数
        t = torch.arange(ft_seq_len, dtype=torch.float32) / ft_seq_len * pt_seq_len

        # freqs: [L, head_dim//2]
        freqs = torch.einsum('l, f -> l f', t, freqs)

        # 每个频率复制两份 -> [L, head_dim]
        freqs = repeat(freqs, 'l n -> l (n r)', r=2)

        # -------- 预先算好 cos / sin，并处理 cls token --------
        if num_cls_token > 0:
            cos = freqs.cos()
            sin = freqs.sin()
            N, D = cos.shape  # N = L, D = head_dim

            # 前面插 num_cls_token 个“零角度”位置：cos=1, sin=0
            cos_pad = torch.ones(num_cls_token, D, dtype=cos.dtype, device=cos.device)
            sin_pad = torch.zeros(num_cls_token, D, dtype=sin.dtype, device=sin.device)

            freqs_cos = torch.cat([cos_pad, cos], dim=0)  # [num_cls + L, head_dim]
            freqs_sin = torch.cat([sin_pad, sin], dim=0)
        else:
            freqs_cos = freqs.cos()  # [L, head_dim]
            freqs_sin = freqs.sin()

        # 不要在这里 .cuda()，用 buffer 让 model.to(device) 自动移动
        self.register_buffer('freqs_cos', freqs_cos, persistent=False)
        self.register_buffer('freqs_sin', freqs_sin, persistent=False)

    def forward(self, t):
        """
        t: (..., seq_len, head_dim)
           比如 [B, heads, L, head_dim] 或 [L, head_dim]
        """
        seq_len = t.shape[-2]

        # 截取对应长度，并对齐到 t 的 device/dtype
        freqs_cos = self.freqs_cos[:seq_len].to(t.device, t.dtype)  # [seq_len, head_dim]
        freqs_sin = self.freqs_sin[:seq_len].to(t.device, t.dtype)

        # broadcast 到 t 的前面几个维度
        while freqs_cos.dim() < t.dim():
            freqs_cos = freqs_cos.unsqueeze(0)
            freqs_sin = freqs_sin.unsqueeze(0)

        return t * freqs_cos + rotate_half(t) * freqs_sin

class RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        LlamaRMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return (self.weight * hidden_states).to(input_dtype)


def get_2d_sincos_pos_embed(embed_dim, grid_size, cls_token=False, extra_tokens=0):
    """
    grid_size: int of the grid height and width
    return:
    pos_embed: [grid_size*grid_size, embed_dim] or [1+grid_size*grid_size, embed_dim] (w/ or w/o cls_token)
    """
    grid_h = np.arange(grid_size, dtype=np.float32)
    grid_w = np.arange(grid_size, dtype=np.float32)
    grid = np.meshgrid(grid_w, grid_h)  # here w goes first
    grid = np.stack(grid, axis=0)

    grid = grid.reshape([2, 1, grid_size, grid_size])
    pos_embed = get_2d_sincos_pos_embed_from_grid(embed_dim, grid)
    if cls_token and extra_tokens > 0:
        pos_embed = np.concatenate([np.zeros([extra_tokens, embed_dim]), pos_embed], axis=0)
    return pos_embed


def get_2d_sincos_pos_embed_from_grid(embed_dim, grid):
    assert embed_dim % 2 == 0

    # use half of dimensions to encode grid_h
    emb_h = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[0])  # (H*W, D/2)
    emb_w = get_1d_sincos_pos_embed_from_grid(embed_dim // 2, grid[1])  # (H*W, D/2)

    emb = np.concatenate([emb_h, emb_w], axis=1) # (H*W, D)
    return emb


def get_1d_sincos_pos_embed_from_grid(embed_dim, pos):
    """
    embed_dim: output dimension for each position
    pos: a list of positions to be encoded: size (M,)
    out: (M, D)
    """
    assert embed_dim % 2 == 0
    omega = np.arange(embed_dim // 2, dtype=np.float64)
    omega /= embed_dim / 2.
    omega = 1. / 10000**omega  # (D/2,)

    pos = pos.reshape(-1)  # (M,)
    out = np.einsum('m,d->md', pos, omega)  # (M, D/2), outer product

    emb_sin = np.sin(out) # (M, D/2)
    emb_cos = np.cos(out) # (M, D/2)

    emb = np.concatenate([emb_sin, emb_cos], axis=1)  # (M, D)
    return emb