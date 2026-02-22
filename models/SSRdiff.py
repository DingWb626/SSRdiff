import torch
import torch.nn as nn
from argparse import Namespace
from models.SSRdiff_diffusion import Diffusion
import torch.nn.functional as F
from models.predictor import MultiLinearModel
from models.dm_layers.decouple import series_decomposition
from models.dm_layers.UIC import UIC
import numpy as np
from models.samplers.dpm_sampler import DPMSolverSampler
from models.S_models import iTransformer ,SVQ ,PatchTST_TS


class SSRdiff(nn.Module):
    def __init__(self, args: Namespace):
        super(SSRdiff, self).__init__()

        self.decom = series_decomposition(args.kernel_size)
        self.input_len = args.input_len
        self.device = args.device
        self.pred_len = args.pred_len
        self.time_steps = args.time_steps
        self.features = args.features

        self.n_z_samples = args.n_z_samples
        self.n_z_samples_depart = args.n_z_samples_depart
        self.DPMsolver_step = args.DPMsolver_step
        self.parameterization = args.parameterization
        self.type_sampler = args.type_sampler
        self.feature_dim = args.feature_dim

        cond_model_dict = {
            'iTransformer': iTransformer,
            'SVQ': SVQ,
            'PatchTST_TS': PatchTST_TS,
        }
        self.cond_pred_model = cond_model_dict[args.cond_model_name].Model(args).float()

        self.diffusion = Diffusion(
            configs=args,
            x_dim=args.feature_dim,
            h_dim=args.h_dim,
            cond_dim=args.feature_dim,
            diff_step_emb_dim=args.diff_dim,
            num_heads=args.n_heads,
            diff_d_state=args.diff_d_state,

            time_steps=args.time_steps,
            feature_dim=args.feature_dim,
            seq_len=args.input_len,
            pred_len=args.pred_len,
            MLP_hidden_dim=args.MLP_hidden_dim,
            emb_dim=args.emb_dim,
            device=self.device,
            beta_scheduler=args.scheduler,
            patch_size=args.patch_size,
        )
        self.sampler = DPMSolverSampler(self.diffusion, self.device, self.parameterization)

        self.seq_len = args.input_len

        self.trend_linear3 = MultiLinearModel(seq_len=args.input_len, pred_len=args.pred_len)

        self.uic = UIC(D=args.feature_dim, d=args.feature_dim, hidden_qk=args.hidden_qk)

        self.mse_loss = nn.MSELoss()
        self.l1_loss = nn.L1Loss()

    def pred(self, x):
        batch_size, input_len, num_features = x.size()

        x_seq = x[:, :self.seq_len, :]
        x_means = x_seq.mean(1, keepdim=True).detach()
        x_enc = x_seq - x_means
        x_stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)

        x_norm = x - x_means
        x_norm /= x_stdev

        x_seq_input = x_norm[:, :self.seq_len, :]
        season_seq, trend_seq = self.decom(x_seq_input)

        x_pred = x_norm[:, -self.pred_len:, :]
        season_pred, trend_pred = self.decom(x_pred)

        trend_seq_pred = self.trend_linear3(trend_seq)
        #  season_seq_pred = self.non_linear(season_seq)
        season_seq_pred,_,_ = self.cond_pred_model(season_seq, None, None, None)
        pred_seq_pred = trend_seq_pred + season_seq_pred

        uic = self.uic(season_seq_pred)

        # Noising Diffusion
        # t = torch.randint(0, self.time_steps, (batch_size,), device=self.device)
        n = x_norm.size(0)
        t = torch.randint(
            low=0, high=self.time_steps, size=(n // 2 + 1,)
        ).to(self.device)

        t = torch.cat([t, self.time_steps - 1 - t], dim=0)[:n].to(self.device)  # t: [batch]

        season_pred1 = season_pred - season_seq_pred

        noise_season = self.diffusion(season_pred1, t)

        season_pred2 = self.diffusion.pred(noise_season, t, uic)

        predict_x = trend_seq_pred + season_pred2 + season_seq_pred

        dec_out = predict_x * \
                  (x_stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (x_means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        trend_loss = self.l1_loss(trend_seq_pred, trend_pred)

        season_loss = self.l1_loss(season_seq_pred, season_pred)

        diffusion_loss = self.mse_loss(season_pred2, season_pred1)

        return dec_out, trend_loss, season_loss, diffusion_loss

    def forecast_vali(self, input_x):
        x = input_x[:, :self.seq_len, :]
        b, _, dim = x.shape

        x_means = x.mean(1, keepdim=True).detach()
        x_enc = x - x_means
        x_stdev = torch.sqrt(
            torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)
        x_enc /= x_stdev
        season, trend = self.decom(x_enc)

        trend_pred_part = self.trend_linear3(trend)
        # season_pred_part = self.non_linear(season)
        season_pred_part,_,_ = self.cond_pred_model(season, None, None, None)

        uic = self.uic(season_pred_part)

        shape = (x_enc.shape[0], self.pred_len, x_enc.shape[-1])
        if self.type_sampler == "DDIM":
            season_pred = self.diffusion.sample_infill(shape, self.time_steps,
                                                       uic)  # [bs*sample, pred_len, var]
        elif self.type_sampler == "DPM_solver":
            season_pred = self.sampler.sample(S=self.DPMsolver_step,
                                              conditioning=uic,
                                              shape=shape,
                                              verbose=False,
                                              unconditional_guidance_scale=1.0,
                                              unconditional_conditioning=None,
                                              eta=0.,
                                              x_T=None)

        predict_x = trend_pred_part + season_pred + season_pred_part

        dec_out = predict_x * \
                  (x_stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))
        dec_out = dec_out + \
                  (x_means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1))

        return dec_out

    def forecast_test(self, input_x):
        x = input_x[:, :self.seq_len, :]  # [bs, seq_len, var]

        x_means = x.mean(1, keepdim=True).detach()  # [bs, 1, var]
        x_enc = x - x_means
        x_stdev = torch.sqrt(torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5)  # [bs, 1, var]
        x_enc /= x_stdev

        def exact_y_0(y_tile_seq):

            return y_tile_seq.reshape(
                -1,
                int(self.n_z_samples / self.n_z_samples_depart),
                self.pred_len,
                self.feature_dim
            )

        season, trend = self.decom(x_enc)
        trend_pred_part = self.trend_linear3(trend)
        season_pred_part,_,_ = self.cond_pred_model(season, None, None, None)
        pred_part = trend_pred_part + season_pred_part
        uic = self.uic(season_pred_part)

        predict_x_box = []

        for _ in range(self.n_z_samples_depart):
            repeat_n = int(self.n_z_samples / self.n_z_samples_depart)

            x_tile = x_enc.repeat(repeat_n, 1, 1, 1)
            x_tile = x_tile.transpose(0, 1).flatten(0, 1).to(self.device)  # [bs*sample, seq_len, var]

            trend_pred_part_tile = trend_pred_part.repeat(repeat_n, 1, 1, 1)
            trend_pred_part_tile = trend_pred_part_tile.transpose(0, 1).flatten(0, 1).to(
                self.device)  # [bs*sample, pred_len, var]

            season_pred_part_tile = season_pred_part.repeat(repeat_n, 1, 1, 1)
            season_pred_part_tile = season_pred_part_tile.transpose(0, 1).flatten(0, 1).to(
                self.device)  # [bs*sample, pred_len, var]

            uic_tile = uic.repeat(repeat_n, 1, 1, 1)
            uic_tile = uic_tile.transpose(0, 1).flatten(0, 1).to(self.device)  # [bs*sample, pred_len, var]

            shape = (x_tile.shape[0], self.pred_len, x_tile.shape[-1])
            if self.type_sampler == "DDIM":
                season_pred = self.diffusion.sample_infill(shape, self.time_steps,
                                                           uic_tile)  # [bs*sample, pred_len, var]
            elif self.type_sampler == "DPM_solver":
                season_pred = self.sampler.sample(S=self.DPMsolver_step,
                                                  conditioning=uic_tile,
                                                  shape=shape,
                                                  verbose=False,
                                                  unconditional_guidance_scale=1.0,
                                                  unconditional_conditioning=None,
                                                  eta=0.,
                                                  x_T=None)

            predict_x = trend_pred_part_tile + season_pred + season_pred_part_tile  # [bs*sample, pred_len, var]

            n_samples = repeat_n

            x_means_expanded = x_means.repeat(n_samples, 1, 1).reshape(-1, 1, self.feature_dim).to(
                self.device)  # [bs*sample, 1, var]
            x_stdev_expanded = x_stdev.repeat(n_samples, 1, 1).reshape(-1, 1, self.feature_dim).to(
                self.device)  # [bs*sample, 1, var]

            predict_x = predict_x * x_stdev_expanded.repeat(1, self.pred_len, 1)
            predict_x = predict_x + x_means_expanded.repeat(1, self.pred_len, 1)

            #  [bs, sample, pred_len, var]
            predict_x = exact_y_0(y_tile_seq=predict_x)
            predict_x_box.append(predict_x.cpu().numpy())

        outputs = np.concatenate(predict_x_box, axis=1)  # [bs, n_samples, pred_len, var]
        f_dim = -1 if self.features == 'MS' else 0
        outputs = outputs[:, :, -self.pred_len:, f_dim:]

        return outputs

    def forward(self, x, task):
        if task == 'train':
            return self.pred(x)
        elif task == 'valid':
            return self.forecast_vali(x)
        elif task == 'test':
            return self.forecast_test(x)
        else:
            raise ValueError(f"Invalid task: {task=}")
