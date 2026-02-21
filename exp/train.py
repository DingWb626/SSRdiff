import torch
import torch.nn as nn
import gc
from pathlib import Path
from dataclasses import dataclass
from tqdm import tqdm
from torch.optim import AdamW
from torch.optim.lr_scheduler import ExponentialLR
from utils.tools import EarlyStopping, EpochTimer
import time
import numpy as np
from multiprocessing import Pool
import CRPS.CRPS as pscore
from utils.metrics import calc_quantile_CRPS
from utils.metrics import calc_quantile_CRPS_sum
from utils.metrics import metric




def ccc(id, pred, true):

    res_box = np.zeros(len(true))

    for i in range(len(true)):
        pred_i = pred[:, i] if pred.ndim == 2 else pred[i]
        true_i = true[i]

        res = pscore(pred_i, true_i).compute()

        # 兼容各种类型的输出
        if isinstance(res, tuple):
            res_val = res[0]
        elif isinstance(res, np.ndarray):
            res_val = res.item() if res.size == 1 else np.mean(res)
        elif isinstance(res, (list,)):
            res_val = float(res[0])
        else:
            res_val = float(res)

        res_box[i] = res_val

    return res_box



def log_normal(x, mu, var):

    eps = 1e-8
    if eps > 0.0:
        var = var + eps
    # return -0.5 * torch.sum(
    #     np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var)
    return 0.5 * torch.mean(
        np.log(2.0 * np.pi) + torch.log(var) + torch.pow(x - mu, 2) / var)


def calculate_crps_sum_worker(args):
    pred, true = args
    p_in = np.sum(pred, axis=-1).T
    t_in = np.sum(true, axis=-1).reshape(-1)
    crps = ccc(8, p_in, t_in)
    return crps.mean()


def calculate_crps_worker(args):
    pred, true = args  # pred: (L, D) 或 (num_samples, L, D)

    if pred.ndim == 2:
        # 单样本预测: [L, D] -> [num_samples=1, L, D]
        p_in = pred[None, :, :]
    elif pred.ndim == 3:
        # 多样本预测
        p_in = pred
    else:
        raise ValueError(f"Unexpected pred shape: {pred.shape}")

    t_in = true  # shape [L, D]
    all_res = []
    for i in range(p_in.shape[-1]):
        # p_in[:, :, i]: [num_samples, L]
        # t_in[:, i]: [L]
        crps = ccc(8, p_in[:, :, i], t_in[:, i])
        all_res.append(crps)

    all_res = np.array(all_res)
    return np.mean(all_res)


@dataclass
class PerformanceMetrics:
    mse_loss: float = 0.0
    mae_loss: float = 0.0
    crps_score: float = 0.0

    def __repr__(self):
        return f"MSE Loss: {self.mse_loss:.6f}, MAE Loss: {self.mae_loss:.6f}, CRPS: {self.crps_score:.6f}"


class ModelTrainer:
    def __init__(self, args, model, device, train_loader, val_loader, test_loader, test_dataset):
        self.args = args
        self.verbose = args.verbose
        self.model = model.to(device)
        self.device = device
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        self.test_dataset = test_dataset

        self.mse_criterion = nn.MSELoss()
        self.mae_criterion = nn.L1Loss()

        self.optimizer = AdamW(self.model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
        self.scheduler = ExponentialLR(optimizer=self.optimizer, gamma=args.lr_decay)

        self.num_epochs = args.num_epochs
        self.eval_freq = args.eval_frequency

        self.save_dir = Path(args.save_dir)
        self.train_log_path = self.save_dir / "train_log.txt"
        self.val_log_path = self.save_dir / "val_log.txt"
        self.test_log_path = self.save_dir / "test_log.txt"
        self.model_save_path = self.save_dir / "model_checkpoint.pth"

        self.early_stopping = EarlyStopping(self.args)
        self.epoch_timer = EpochTimer()

        self.n_blocks = args.n_blocks
        self.rmom_n = args.rmom
        self.n_bins = args.n_bins
        self.PICP_range = args.PICP_range


    def train(self):
        self.train_log_path.write_text("")
        self.val_log_path.write_text("")

        for epoch in range(self.num_epochs):
            self.epoch_timer.start()
            train_metrics, train_speed = self._train_one_epoch()

            with self.train_log_path.open("a") as log_file:
                log_file.write(f"Epoch {epoch + 1}: {train_metrics} | Speed: {train_speed:.2f} it/s\n")

            if self.verbose:
                print(f"Training Epoch {epoch + 1}: {train_metrics} | Speed: {train_speed:.2f} it/s")

            self.epoch_timer.stop()
            if self.verbose:
                self.epoch_timer.print_duration(epoch=epoch + 1, total_epochs=self.num_epochs)

            if (epoch + 1) % self.eval_freq == 0:
                val_metrics, val_speed = self._validate_one_epoch()
                with self.val_log_path.open("a") as log_file:
                    log_file.write(f"Epoch {epoch + 1}: {val_metrics} | Speed: {val_speed:.2f} it/s\n")

                if self.verbose:
                    print(f"Validation Epoch {epoch + 1}: {val_metrics} | Speed: {val_speed:.2f} it/s")

                self.early_stopping(val_metrics.mse_loss, self.model, self.model_save_path)
                if self.early_stopping.early_stop:
                    if self.verbose:
                        print("Early stopping triggered.")
                    break

    def _train_one_epoch(self):
        self.model.train()
        metrics = PerformanceMetrics()
        total_iters, total_time = 0, 0

        for x, y in tqdm(self.train_loader, desc="Training", disable=not self.args.use_tqdm):
            start_time = time.time()
            x, y = x.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()

            predictions, trend_loss, season_loss, diffusion_loss = self.model(x, task="train")
            mse_loss1 = self.mse_criterion(predictions, y)
            mae_loss1 = self.mae_criterion(predictions, y)
            mse_loss = mse_loss1 + 0.3 * season_loss + 0.3 * trend_loss + 0.4 * diffusion_loss
            mae_loss = mae_loss1 + 0.3 * season_loss + 0.3 * trend_loss + 0.4 * diffusion_loss

            # --- 反向传播 ---
            mse_loss.backward()
            self.optimizer.step()

            total_time += time.time() - start_time
            total_iters += 1
            metrics.mse_loss += mse_loss.item()
            metrics.mae_loss += mae_loss.item()

        self.scheduler.step()

        metrics.mse_loss /= len(self.train_loader)
        metrics.mae_loss /= len(self.train_loader)

        avg_iter_speed = total_iters / total_time
        return metrics, avg_iter_speed

    @torch.no_grad()
    def _validate_one_epoch(self):
        self.model.eval()
        metrics = PerformanceMetrics()
        total_iters, total_time = 0, 0

        for x, y in tqdm(self.val_loader, desc="Validation", disable=not self.args.use_tqdm):
            start_time = time.time()
            x, y = x.to(self.device), y.to(self.device)

            predictions = self.model(x, task="valid")

            mse_loss = self.mse_criterion(predictions, y)
            mae_loss = self.mae_criterion(predictions, y)

            total_time += time.time() - start_time
            total_iters += 1
            metrics.mse_loss += mse_loss.item()
            metrics.mae_loss += mae_loss.item()


        metrics.mse_loss /= len(self.val_loader)
        metrics.mae_loss /= len(self.val_loader)
        avg_iter_speed = total_iters / total_time
        return metrics, avg_iter_speed


    def calculate_batch_crps(self, pred, true):

        pool = Pool(processes=32)
        crps_values = pool.map(calculate_crps_worker, zip(pred, true))

        # crps_sum_values = pool.map(calculate_crps_sum_worker, zip(pred, true))
        pool.close()
        pool.join()
        # print('1',len(crps_values))
        # print('2',len(crps_sum_values))
        # 累加所有样本的 CRPS
        batch_crps = np.sum(crps_values)
        # batch_crps_sum = np.sum(crps_sum_values)
        return batch_crps


    @torch.no_grad()
    def evaluate_test(self):
        self.test_log_path.write_text("")
        self.model.load_state_dict(torch.load(self.model_save_path, weights_only=True))
        self.model.eval()


        predictions_s = []
        y_s = []
        metrics = PerformanceMetrics()
        total_mse = 0.0
        total_mae = 0.0
        total_samples = 0.0
        sum_crps = 0.0
        i=1

        for x, y in tqdm(self.test_loader, desc="Testing", disable=not self.args.use_tqdm):
            x, y = x.to(self.device), y.to(self.device)
            predictions = self.model(x, task="test")
            B, pred_len, _, _ = predictions.shape

            y = y.detach().cpu().numpy()
            print(f"{i}",y.shape,predictions.shape)

            crps_score = self.calculate_batch_crps(predictions, y)
            sum_crps += crps_score


            predictions_mean = np.mean(predictions, axis=1)   #(32, 192, 7)
            mae, mse, rmse, mape, mspe = metric(predictions_mean, y)
            total_mse += mse * predictions_mean.shape[0]
            total_mae += mae * predictions_mean.shape[0]
            total_samples += predictions_mean.shape[0]
            predictions_s.append(predictions.sum(-1))
            y_s.append(y.sum(-1))
            i+=1
            del predictions
            gc.collect()


        avg_crps = sum_crps / total_samples

        mse_total = total_mse / total_samples
        mae_total = total_mae / total_samples
        print('NT metrc: CRPS:{:.4f}'.format(avg_crps))
        print('NT metrc: mse:{:.4f}, mae:{:.4f} '.format(mse_total, mae_total))
        predictions_s = np.concatenate(predictions_s, axis=0)
        y_s = np.concatenate(y_s, axis=0)
        crps_sum=calc_quantile_CRPS_sum(predictions_s,y_s)
        crps=calc_quantile_CRPS(predictions_s,y_s)
        print('NT metrc: CRPS_sum:{:.4f}'.format(crps_sum))


        metrics.mse_loss = mse_total
        metrics.mae_loss = mae_total
        metrics.crps_score = avg_crps


        with self.test_log_path.open("w") as log_file:
            log_file.write(f"{metrics}\n")

        return metrics
