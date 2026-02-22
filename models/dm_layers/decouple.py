import math
import torch
import torch.nn as nn
from einops import rearrange, reduce, repeat


def split_high_low_freq_torch(x, epsilon=0.9, per_channel=True):
    """
    Args:
        x: torch.Tensor, shape [B, L, C] (real)
        epsilon: float in (0,1)
        per_channel: bool

    Returns:
        x_low, x_high, kappa
        - x_low/x_high: [B, L, C]
        - kappa: [B, C] if per_channel else [B]
    """
    if x.dim() != 3:
        raise ValueError("Expected x shape [B, L, C]")

    B, L, C = x.shape
    X = torch.fft.rfft(x, dim=1)  # [B, F, C], complex
    F = X.size(1)

    E = (X.real ** 2 + X.imag ** 2)  # [B, F, C]

    if per_channel:
        cumE = torch.cumsum(E, dim=1)
        totalE = E.sum(dim=1, keepdim=True).clamp_min(1e-12)
        ratio = cumE / totalE  # [B, F, C]
        # first freq index where ratio >= epsilon
        # torch doesn't have argfirst; we can do a masked argmax trick
        mask = (ratio >= epsilon).to(torch.int64)  # [B, F, C]
        # make sure at least one True; if none, kappa=F
        has = mask.any(dim=1)  # [B, C]
        first = mask.argmax(dim=1) + 1  # [B, C] (count of bins)
        kappa = torch.where(has, first, torch.full_like(first, F))
    else:
        E_agg = E.sum(dim=2)  # [B, F]
        cumE = torch.cumsum(E_agg, dim=1)
        totalE = E_agg.sum(dim=1, keepdim=True).clamp_min(1e-12)
        ratio = cumE / totalE  # [B, F]
        mask = (ratio >= epsilon).to(torch.int64)  # [B, F]
        has = mask.any(dim=1)  # [B]
        first = mask.argmax(dim=1) + 1  # [B]
        kappa = torch.where(has, first, torch.full_like(first, F))

    X_low = torch.zeros_like(X)
    X_high = torch.zeros_like(X)

    if per_channel:
        # vectorized-ish: loop over C is usually fine; F is small relative to L
        for c in range(C):
            k = kappa[:, c]  # [B]
            for b in range(B):
                kk = int(k[b].item())
                X_low[b, :kk, c] = X[b, :kk, c]
                X_high[b, kk:, c] = X[b, kk:, c]
    else:
        for b in range(B):
            kk = int(kappa[b].item())
            X_low[b, :kk, :] = X[b, :kk, :]
            X_high[b, kk:, :] = X[b, kk:, :]

    x_low = torch.fft.irfft(X_low, n=L, dim=1)
    x_high = torch.fft.irfft(X_high, n=L, dim=1)
    return x_high, x_low, kappa

class moving_avg(nn.Module):

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        front = x[:, 0:1, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        end = x[:, -1:, :].repeat(1, (self.kernel_size - 1) // 2, 1)
        x = torch.cat([front, x, end], dim=1)
        x = self.avg(x.permute(0, 2, 1))
        x = x.permute(0, 2, 1)
        return x


class series_decomposition(nn.Module):

    def __init__(self, kernel_size):
        super(series_decomposition, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class EMA(nn.Module):
    """
    Exponential Moving Average (EMA) block to highlight the trend of time series
    """

    def __init__(self, alpha):
        super(EMA, self).__init__()
        # self.alpha = nn.Parameter(alpha)    # Learnable alpha
        self.alpha = alpha

    # Optimized implementation with O(1) time complexity
    def forward(self, x):
        # x: [Batch, Input, Channel]
        # self.alpha.data.clamp_(0, 1)        # Clamp learnable alpha to [0, 1]
        _, t, _ = x.shape
        device = x.device
        powers = torch.flip(torch.arange(t, dtype=torch.double, device=device), dims=(0,))
        weights = torch.pow((1 - self.alpha), powers)
        divisor = weights.clone()
        weights[1:] = weights[1:] * self.alpha
        weights = weights.reshape(1, t, 1)
        divisor = divisor.reshape(1, t, 1)
        x = torch.cumsum(x * weights, dim=1)
        x = torch.div(x, divisor)
        return x.to(torch.float32)


class series_decomposition1(nn.Module):

    def __init__(self, alpha):
        super(series_decomposition1, self).__init__()
        self.ma = EMA(alpha)

    def forward(self, x):
        moving_average = self.ma(x)
        res = x - moving_average
        return res, moving_average


def main_freq_part(x, k, rfft=True):
    # freq normalization
    # start = time.time()
    if rfft:
        xf = torch.fft.rfft(x, dim=1)
    else:
        xf = torch.fft.fft(x, dim=1)

    k_values = torch.topk(xf.abs(), k, dim=1)
    indices = k_values.indices

    mask = torch.zeros_like(xf)
    mask.scatter_(1, indices, 1)
    xf_filtered = xf * mask

    if rfft:
        x_filtered = torch.fft.irfft(xf_filtered, dim=1).real.float()
    else:
        x_filtered = torch.fft.ifft(xf_filtered, dim=1).real.float()

    norm_input = x - x_filtered
    # print(f"decompose take:{ time.time() - start} s")
    return norm_input, x_filtered


class FourierLayer(nn.Module):
    """
    Model seasonality of time series using the inverse DFT.
    """

    def __init__(self,  low_freq=1, factor=1):
        super().__init__()
        self.factor = factor
        self.low_freq = low_freq

    def forward(self, x):
        """x: (b, t, d)"""
        b, t, d = x.shape
        x_freq = torch.fft.rfft(x, dim=1)

        if t % 2 == 0:
            x_freq = x_freq[:, self.low_freq:-1]
            f = torch.fft.rfftfreq(t)[self.low_freq:-1]
        else:
            x_freq = x_freq[:, self.low_freq:]
            f = torch.fft.rfftfreq(t)[self.low_freq:]

        x_freq, index_tuple = self.topk_freq(x_freq)
        f = repeat(f, 'f -> b f d', b=x_freq.size(0), d=x_freq.size(2)).to(x_freq.device)
        f = rearrange(f[index_tuple], 'b f d -> b f () d').to(x_freq.device)
        return self.extrapolate(x_freq, f, t)

    def extrapolate(self, x_freq, f, t_0):
        x_freq = torch.cat([x_freq, x_freq.conj()], dim=1)
        f = torch.cat([f, -f], dim=1)
        t = rearrange(torch.arange(t_0, dtype=torch.float),
                      't -> () () t ()').to(x_freq.device)

        amp = x_freq.abs() * 2 / t_0
        amp = rearrange(amp, 'b f d -> b f () d')
        phase = rearrange(x_freq.angle(), 'b f d -> b f () d')
        x_time = amp * torch.cos(2 * math.pi * f * t + phase)
        return reduce(x_time, 'b f t d -> b t d', 'sum')

    def topk_freq(self, x_freq):
        length = x_freq.shape[1]
        top_k = int(self.factor * math.log(length))  # 计算 top_k 的值
        # torch.topk
        #   largest: if True，按照大到小排序； if False，按照小到大排序
        #   sorted: 返回的结果按照顺序返回
        values, indices = torch.topk(x_freq.abs(), top_k, dim=1, largest=True, sorted=True)
        mesh_a, mesh_b = torch.meshgrid(torch.arange(x_freq.size(0)), torch.arange(x_freq.size(2)), indexing='ij')
        index_tuple = (mesh_a.unsqueeze(1), indices, mesh_b.unsqueeze(1))
        x_freq = x_freq[index_tuple]
        return x_freq, index_tuple