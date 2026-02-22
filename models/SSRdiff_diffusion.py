import torch
import torch.nn as nn
from models.denoise_network.networks import AttentionNet
from functools import partial


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1,) * (len(x_shape) - 1)))


class Diffusion(nn.Module):
    def __init__(
            self,
            configs,
            x_dim: int,
            h_dim: int,
            cond_dim: int,
            diff_step_emb_dim: int,
            num_heads: int,
            diff_d_state: int,

            time_steps: int,
            feature_dim: int,
            seq_len: int,
            pred_len: int,
            MLP_hidden_dim: int,
            emb_dim: int,
            patch_size: int,
            device: torch.device,
            beta_scheduler: str = "cosine",

    ):
        super(Diffusion, self).__init__()
        self.device = device
        self.time_steps = time_steps
        self.seq_length = seq_len
        self.pred_length = pred_len
        self.args = configs

        if beta_scheduler == 'cosine':
            self.betas = self._cosine_beta_schedule().to(self.device)
        elif beta_scheduler == 'linear':
            self.betas = self._linear_beta_schedule().to(self.device)
        elif beta_scheduler == 'exponential':
            self.betas = self._exponential_beta_schedule().to(self.device)
        elif beta_scheduler == 'inverse_sqrt':
            self.betas = self._inverse_sqrt_beta_schedule().to(self.device)
        elif beta_scheduler == 'piecewise':
            self.betas = self._piecewise_beta_schedule().to(self.device)
        else:
            raise ValueError(f"Unknown schedule type: {scheduler}")

        self.eta = 0
        self.alpha = 1 - self.betas
        self.alphas_cumprod = torch.cumprod(self.alpha, dim=0)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1. / self.alphas_cumprod - 1)
        self.gamma = torch.cumprod(self.alpha, dim=0).to(self.device)

        self.denoise_net = AttentionNet(x_dim, h_dim, cond_dim, diff_step_emb_dim, diff_d_state, num_heads, 0 ,self.pred_length)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
                extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise
        )

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
                extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def _cosine_beta_schedule(self, s=0.008):
        steps = self.time_steps + 1
        x = torch.linspace(0, self.time_steps, steps)
        alpha_cumprod = (
                torch.cos(((x / self.time_steps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        )
        alpha_cumprod = alpha_cumprod / alpha_cumprod[0]
        betas = 1 - (alpha_cumprod[1:] / alpha_cumprod[:-1])
        return torch.clip(betas, 0, 0.999)

    def _linear_beta_schedule(self, beta_start=1e-4, beta_end=0.02):
        return torch.linspace(beta_start, beta_end, self.time_steps)

    def _exponential_beta_schedule(self, beta_start=1e-4, beta_end=0.02):
        steps = self.time_steps
        return beta_start * ((beta_end / beta_start) ** (torch.linspace(0, 1, steps)))

    def _inverse_sqrt_beta_schedule(self, beta_start=1e-4):
        steps = self.time_steps
        x = torch.arange(1, steps + 1)
        return torch.clip(beta_start / torch.sqrt(x), 0, 0.999)

    def _piecewise_beta_schedule(self, beta_values=[1e-4, 0.01, 0.02], segment_steps=[100, 200, 300]):
        assert len(beta_values) == len(segment_steps), "beta_values and segment_steps length mismatch"
        betas = [torch.full((steps,), beta) for beta, steps in zip(beta_values, segment_steps)]
        return torch.cat(betas)[:self.time_steps]

    def extract(self, a, t, x_shape):
        b, *_ = t.shape
        t = t.to(a.device)
        out = a.gather(-1, t)
        return out.reshape(b, *((1,) * (len(x_shape) - 1)))

    def noise(self, x, t):
        noise = torch.randn_like(x)
        gamma_t = self.gamma[t].unsqueeze(-1).unsqueeze(-1)
        noisy_x = torch.sqrt(gamma_t) * x + torch.sqrt(1 - gamma_t) * noise
        return noisy_x, noise

    def forward(self, x, t):
        noisy_x, _ = self.noise(x, t)
        return noisy_x

    def pred(self, x, t, cond):
        if t == None:
            t = torch.randint(0, self.time_steps, (x.shape[0],), device=self.device)
        return self.denoise_net(x, t, cond)

    def predict_noise_from_start(self, x_t, t, x0):
        return (
                (self.extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) /
                self.extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def model_predictions(self, x, t, cond, clip_x_start=False, padding_masks=None):

        maybe_clip = partial(torch.clamp, min=-1., max=1.) if clip_x_start else identity
        x_start = self.denoise_net(x, t, cond)
        x_start = maybe_clip(x_start)
        pred_noise = self.predict_noise_from_start(x, t, x_start)
        return pred_noise, x_start

    @torch.no_grad()
    def sample_infill(self, shape, sampling_timesteps, cond, clip_denoised=True):

        if isinstance(shape, torch.Tensor):
            batch_size, _, _ = shape.shape
        else:
            batch_size, _, _ = shape

        batch, device, total_timesteps, eta = shape[0], self.device, self.time_steps, self.eta

        times = torch.linspace(-1, total_timesteps - 1, steps=sampling_timesteps + 1)

        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))

        if isinstance(shape, torch.Tensor):
            denoise_series = torch.randn(shape.shape, device=device)
        else:
            denoise_series = torch.randn(shape, device=device)

        for time, time_next in time_pairs:
            time_cond = torch.full((batch_size,), time, device=device, dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(denoise_series, time_cond, cond,
                                                             clip_x_start=clip_denoised)

            if time_next < 0:
                denoise_series = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]
            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()
            pred_mean = x_start * alpha_next.sqrt() + c * pred_noise
            noise = torch.randn_like(denoise_series)

            denoise_series = pred_mean + sigma * noise

        return denoise_series

    @torch.no_grad()
    def ddim_sample(self, shape, sampling_timesteps, cond, pred_type, eta=0., clip_denoised=True):

        # --- 安全处理 shape ---
        if isinstance(shape, torch.Tensor):
            shape = shape.flatten().tolist()
        elif isinstance(shape, torch.Size):
            shape = list(shape)
        elif isinstance(shape, (list, tuple)):
            shape = [int(x.item()) if torch.is_tensor(x) else int(x) for x in shape]
        else:
            raise TypeError(f"Unsupported shape type: {type(shape)}")

        shape = tuple(shape)

        batch, device, total_timesteps = shape[0], self.betas.device, self.time_steps

        times = torch.linspace(-1, total_timesteps - 1,
                               steps=sampling_timesteps + 1)  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(zip(times[:-1], times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=device)

        x_start = None

        for time, time_next in time_pairs:
            time_cond = torch.full((batch,), time, device=device, dtype=torch.long)
            if pred_type == "data":
                pred_noise = self.denoise_net(img, time_cond, cond)
                pred_noise = self.predict_noise_from_start(img, time_cond, x_start)
            elif pred_type == "noise":
                pred_noise = self.denoise_net(img, time_cond, cond)
                x_start = self.predict_start_from_noise(img, time_cond, pred_noise)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) / (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma ** 2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        return img
