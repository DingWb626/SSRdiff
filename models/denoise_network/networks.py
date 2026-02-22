import torch
import numpy as np
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from inspect import isfunction
from mamba_ssm import Mamba
import math
from utils.masking import generate_causal_mask



def exists(val):
    return val is not None


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


class TransformerEncoderBlock(nn.Module):
    def __init__(
        self, d_model: int, num_heads: int, feedforward_dim: int, dropout: float
    ):
        super(TransformerEncoderBlock, self).__init__()

        self.attention = nn.MultiheadAttention(
            embed_dim=d_model, num_heads=num_heads, dropout=dropout, batch_first=True
        )
        self.norm1 = nn.LayerNorm(d_model)
        self.ff = nn.Sequential(
            nn.Linear(d_model, feedforward_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(feedforward_dim, d_model),
        )
        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=feedforward_dim, kernel_size=1)
        self.activation = nn.GELU()
        self.conv2 = nn.Conv1d(in_channels=feedforward_dim, out_channels=d_model, kernel_size=1)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, mask):
        """
        :param x: [batch_size * num_features, seq_len, d_model]
        :param mask: [1, 1, seq_len, seq_len]
        :return: [batch_size * num_features, seq_len, d_model]
        """
        # Self-attention
        attn_output, _ = self.attention(x, x, x, attn_mask=mask)
        x = self.norm1(x + self.dropout(attn_output))

        # Feed-forward network
        ff_output = self.ff(x)
        output = self.norm2(x + self.dropout(ff_output))

        return output


class CausalTransformer(nn.Module):
    def __init__(
        self,
        d_model: int,
        num_heads: int,
        feedforward_dim: int,
        dropout: float,
    ):
        super(CausalTransformer, self).__init__()

        self.layer = TransformerEncoderBlock(d_model, num_heads, feedforward_dim, dropout)
        
        self.norm = nn.LayerNorm(d_model)

    def forward(self, x, is_mask=True):
        # x: [batch_size * num_features, seq_len, d_model]
        seq_len = x.size(1)
        mask = generate_causal_mask(seq_len).to(x.device) if is_mask else None
        x = self.layer(x, mask)
        x = self.norm(x)
        return x

class LinearEncoder(nn.Module):
    def __init__(self, d_model, d_ff=None, CovMat=None, dropout=0.1, activation="relu", token_num=None, **kwargs):
        super(LinearEncoder, self).__init__()

        d_ff = d_ff or 4 * d_model
        self.d_model = d_model
        self.d_ff = d_ff
        self.CovMat = CovMat.unsqueeze(0) if CovMat is not None else None
        self.token_num = token_num

        self.norm1 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        # attention --> linear
        self.v_proj = nn.Linear(d_model, d_model)
        self.out_proj = nn.Linear(d_model, d_model)

        init_weight_mat = torch.eye(self.token_num) * 1.0 + torch.randn(self.token_num, self.token_num) * 1.0  #(token_num, token_num)
        self.weight_mat = nn.Parameter(init_weight_mat[None, :, :])  #(1, token_num, token_num)

        # self.bias = nn.Parameter(torch.zeros(1, 1, self.d_model))

        self.conv1 = nn.Conv1d(in_channels=d_model, out_channels=d_ff, kernel_size=1)
        self.conv2 = nn.Conv1d(in_channels=d_ff, out_channels=d_model, kernel_size=1)
        self.activation = F.relu if activation == "relu" else F.gelu
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, x, **kwargs):
        # x.shape: b, l, d_model
        values = self.v_proj(x)

        if self.CovMat is not None:
            A = F.softmax(self.CovMat, dim=-1) + F.softplus(self.weight_mat)
        else:
            A = F.softplus(self.weight_mat)

        A = F.normalize(A, p=1, dim=-1)
        A = self.dropout(A)

        new_x = A @ values  # + self.bias      #(1,L,L)@(B,L,dmodel)⇒(B,L,dmodel)

        x = x + self.dropout(self.out_proj(new_x))
        x = self.norm1(x)

        y = self.dropout(self.activation(self.conv1(x.transpose(-1, 1))))
        y = self.dropout(self.conv2(y).transpose(-1, 1))
        output = self.norm2(x + y)

        return output

class SelfAttention(nn.Module):
    def __init__(self, x_dim, heads=4, dim_head=64, dropout=0.):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv1d(x_dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv1d(hidden_dim, x_dim, 1)

    def forward(self, x):  # x: [B, x_dim, N]
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(lambda t: rearrange(t, 'b (h c) n -> b h c n', h=self.heads), qkv)
        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h n d -> b (h d) n')
        return self.to_out(out)  # out: [B, x_dim, N]


class CrossAttention(nn.Module):
    def __init__(self, x_dim, cond_dim, num_heads, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * num_heads

        self.scale = dim_head ** -0.5
        self.heads = num_heads

        self.to_q = nn.Linear(x_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(cond_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(cond_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, x_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, cond):
        h = self.heads
        # print(x.shape, cond.shape)
        q = self.to_q(x)  # [B, L_x, inner_dim]
        k = self.to_k(cond)
        v = self.to_v(cond)

        q, k, v = map(lambda t: rearrange(t, 'b l (h d) -> b h l d', h=h), (q, k, v))  # [B, H, L_x, d]
        q = q * self.scale

        sim = einsum('b h i d, b h j d -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h l d -> b l (h d)', h=self.heads)
        return self.to_out(out)


class FeatureAttention(nn.Module):
    def __init__(self, x_dim, out_dim, num_heads, dim_head, dropout=0.):
        super(FeatureAttention, self).__init__()
        #         # self.token = nn.Parameter(torch.randn(1, tk_dim))
        #         # self.atten_tk = CrossAttention(tk_dim, x_dim, num_heads, dropout=dropout)
        self.atten_x = SelfAttention(x_dim, num_heads, dim_head, dropout=dropout)
        #         # self.ffd_tk = nn.Linear(tk_dim, tk_dim)
        #         # self.norm_tk = nn.LayerNorm(tk_dim)
        self.ffd_x = nn.Linear(x_dim, x_dim)
        self.norm_x = nn.LayerNorm(x_dim)
        self.out_proj = nn.Sequential(
            nn.Linear(x_dim, out_dim),
            nn.LeakyReLU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: [batch_size, in_dim, 1]
        out = self.atten_x(x.unsqueeze(-1))
        out = self.norm_x(out.squeeze(-1) + x)
        out = self.out_proj(out)

        return out


class ResidualBlock(nn.Module):
    def __init__(self, in_dim, diff_dim, cond_dim, res_dim, diff_d_state, num_heads, dropout=0, pre_len: int = 192):
        super(ResidualBlock, self).__init__()

        self.diffproj = nn.Linear(diff_dim, 2 * res_dim)
        self.condproj = nn.Linear(cond_dim, 2 * res_dim)
      #  self.selfattention = SelfAttention(in_dim, num_heads, dropout=dropout)
        self.condattention = CrossAttention(in_dim, 2 * res_dim, num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(in_dim)
        self.norm2 = nn.LayerNorm(in_dim)
        self.outproj = nn.Linear(res_dim, 2 * in_dim)
    #    self.attention = Mamba(in_dim, diff_d_state, d_conv=2, expand=2)
    #    self.attention_r = Mamba(in_dim, diff_d_state, d_conv=2, expand=2)
        self.encoder =LinearEncoder( d_model=in_dim,  CovMat=None, dropout=0.1, activation='gelu', token_num=pre_len)
    #    self.Causalencoder = CausalTransformer(
    #        d_model=in_dim,
    #        num_heads=num_heads,
    #        feedforward_dim=1024,
    #        dropout=dropout,
    #    )

    def forward(self, x, conds, diff_emb):
        # x: [B,L,C]   conds: [B,L,C] c=in_dim
        res = x
        x = x.permute(0, 2, 1)  # B,C,L
        diff_proj = self.diffproj(diff_emb).unsqueeze(-1)  # B,C,1
        x = x + diff_proj
        # print(x.shape, diff_proj.shape)
        # torch.Size([32, 128, 96]) torch.Size([32, 128, 1])
        x = x.permute(0, 2, 1)  # B,L,C
       # x = self.attention(x) + self.attention_r(x.flip(dims=[1])).flip(dims=[1])

        x = self.encoder(x)
       # x = self.Causalencoder(x)

       # x = self.selfattention(x).permute(0, 2, 1)
        x = self.norm1(x)
        cond = self.condproj(conds)
        x = self.condattention(x, cond)
        x = self.norm2(x)

        gate, filt = torch.chunk(x, 2, dim=2)

        x = torch.tanh(gate) * torch.sigmoid(filt)
        # print("outproj input shape:", x.shape)
        # print("outproj weight shape:", self.outproj.weight.shape)

        x = self.outproj(x)
        x = F.leaky_relu(x, 0.4)
        out, skip = torch.chunk(x, 2, dim=2)
        out = (res + out) / torch.sqrt(torch.tensor(2.0))

        return out, skip


class TimestepEmbedder(nn.Module):# 输入: [B], 输出: [B, hidden_size]
    """
    Embeds scalar timesteps into vector representations.
    """
    def __init__(self, hidden_size, frequency_embedding_size=256):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(frequency_embedding_size, hidden_size, bias=True),
            nn.SiLU(),
            nn.Linear(hidden_size, hidden_size, bias=True),
        )
        self.frequency_embedding_size = frequency_embedding_size

    @staticmethod
    def timestep_embedding(t, dim, max_period=10000):
        """
        Create sinusoidal timestep embeddings.
        :param t: a 1-D Tensor of N indices, one per batch element.
                          These may be fractional.
        :param dim: the dimension of the output.
        :param max_period: controls the minimum frequency of the embeddings.
        :return: an (N, D) Tensor of positional embeddings.
        """
        # https://github.com/openai/glide-text2im/blob/main/glide_text2im/nn.py
        half = dim // 2
        freqs = torch.exp(
            -math.log(max_period) * torch.arange(start=0, end=half, dtype=torch.float32) / half
        ).to(device=t.device)
        args = t[:, None].float() * freqs[None]
        embedding = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
        if dim % 2:
            embedding = torch.cat([embedding, torch.zeros_like(embedding[:, :1])], dim=-1)
        return embedding

    def forward(self, t):
        t_freq = self.timestep_embedding(t, self.frequency_embedding_size)
        t_emb = self.mlp(t_freq)
        return t_emb


class AttentionNet(nn.Module):

    def __init__(self, x_dim, h_dim, cond_dim, diff_step_emb_dim, diff_d_state, num_heads, dropout=0, pre_len: int = 192):
        super(AttentionNet, self).__init__()

        self.diff_emb = TimestepEmbedder(h_dim)

        self.input_proj = nn.Sequential(
            nn.Linear(x_dim, h_dim),
            nn.LeakyReLU()
        )

        self.residual_layers = nn.ModuleList(
            [
                ResidualBlock(
                    in_dim=h_dim,
                    diff_dim=h_dim,
                    cond_dim=cond_dim,
                    res_dim=h_dim // 2,
                    diff_d_state=diff_d_state,
                    num_heads=num_heads,
                    dropout=dropout,
                    pre_len=pre_len
                )
            ]
        )

        self.skip_proj = nn.Linear(h_dim, h_dim)
        self.out_proj = nn.Linear(h_dim, x_dim)

    def forward(self, x, diff_t, conds):
        # x: [batch_size, pred_len, in_dim]
        # conds: [batch_size, pred_len, cond_dim]
        # diff_step: [batch_size, diff_step_emb_dim]
        x = self.input_proj(x)
        diff_emb = self.diff_emb(diff_t)
        skip = []
        for layer in self.residual_layers:
            x, s = layer(x, conds, diff_emb)
            skip.append(s)

        x = torch.sum(torch.stack(skip), dim=0) / torch.sqrt(torch.tensor(len(self.residual_layers)))

        x = self.skip_proj(x)
        x = F.leaky_relu(x, 0.4)
        x = self.out_proj(x)

        return x
