import argparse
import torch
import json
from pathlib import Path


def organize_file_structure(_args: argparse.Namespace):
    save_dir = Path(_args.save_dir)
    save_dir = save_dir / _args.dataset
    save_dir = save_dir / _args.model_name
    save_dir = save_dir / str(_args.train_times)
    train_settings = "i{}_o{}".format(
        _args.input_len,
        _args.pred_len
    )

    exp_save_dir = save_dir / train_settings
    exp_save_dir.mkdir(
        parents=True, exist_ok=True
    )
    args_save_path = exp_save_dir / "args.json"
    with open(args_save_path, "w") as f:
        json.dump(vars(_args), f, indent=4)
    scores_save_path = exp_save_dir / "scores.txt"
    _args.scores_save_path = scores_save_path

    log_dir = save_dir / "logs"
    _args.log_dir = log_dir
    log_dir.mkdir(
        parents=True, exist_ok=True
    )

    _args.train_settings = train_settings
    train_save_dir = exp_save_dir / "train"
    _args.train_save_dir = train_save_dir

    if _args.task_name == "train":
        train_save_dir.mkdir(parents=True, exist_ok=True)


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CDPM Args")

    # basic config
    parser.add_argument("--data_dir", type=str, default="datasets", help="data directory")
    parser.add_argument("--dataset", type=str, default="ETTh1", help="dataset name")
    parser.add_argument("--save_dir", type=str, default="results", help="save results or train models directory")

    # data loader
    parser.add_argument("--train_batch_size", type=int, default=64, help="batch size for training")
    parser.add_argument("--val_batch_size", type=int, default=64, help="batch size for validation")
    parser.add_argument("--test_batch_size", type=int, default=64, help="batch size for testing")
    parser.add_argument("--scale", action="store_true", help="scale data", default=True)
    parser.add_argument("--input_len", type=int, default=96, help="input sequence length")
    parser.add_argument("--pred_len", type=int, default=192, help="predicted sequence length")
    parser.add_argument("--feature_dim", type=int, default=7, help="number of features")
    parser.add_argument("--plot_example", type=bool, default=True, help="")
    parser.add_argument('--features', type=str, default='M',
                        help='forecasting task, options:[M, S, MS]; M:multivariate predict multivariate, S:univariate predict univariate, MS:multivariate predict univariate')

    # model define
    parser.add_argument("--model_name", type=str, default="SSRdiff", help="model name")
    parser.add_argument("--time_steps", type=int, default=100, help="time steps in diffusion")
    parser.add_argument("--scheduler", type=str, default="cosine", help="scheduler in diffusion")
    parser.add_argument("--MLP_hidden_dim", type=int, default=256, help="MLP hidden dim")
    parser.add_argument("--emb_dim", type=int, default=256, help="emb dim")
    parser.add_argument("--hidden_qk", type=int, default=64, help="hidden_qk")
    parser.add_argument("--n_z_samples", type=int, default=100, help="")
    parser.add_argument("--n_z_samples_depart", type=int, default=1, help="")
    parser.add_argument('--DPMsolver_step', type=int, default=20, help='')
    parser.add_argument('--parameterization', type=str, default="x_start", help='x_start/noise')
    parser.add_argument('--type_sampler', type=str, default='DPM_solver', help='DPM_solver/DDIM')

    # decomposition
    parser.add_argument('--decomposition', action='store_true', help='decomposition')
    parser.add_argument('--kernel_size', type=int, default=5, help='kernel_size length')
    parser.add_argument('--fourier_factor', type=float, default=1.0, help='factor in computing `top_k`')


    # condition_models(iTransformer)
    parser.add_argument('--cond_model_name', type=str, default='iTransformer', help='SVQ/iTransformer/PatchTST_TS')
    parser.add_argument('--output_attention', action='store_true', help='whether to output attention in ecoder')
    parser.add_argument('--use_norm', type=int, default=True, help='use norm and denorm')
    parser.add_argument('--d_model_i', type=int, default=256, help='dimension of model')
    parser.add_argument('--dropout', type=float, default=0.1, help='dropout')
    parser.add_argument('--embed', type=str, default='timeF',
                        help='time features encoding, options:[timeF, fixed, learned]')
    parser.add_argument('--freq', type=str, default='h',
                        help='freq for time features encoding, options:[s:secondly, t:minutely, h:hourly, d:daily, b:business days, w:weekly, m:monthly], you can also use more detailed freq like 15min or 3h')
    parser.add_argument('--class_strategy', type=str, default='projection', help='projection/average/cls_token')
    parser.add_argument('--activation', type=str, default='gelu', help='activation')
    parser.add_argument('--do_predict', action='store_true', help='whether to predict unseen future data')
    parser.add_argument('--n_heads_i', type=int, default=8, help='num of heads')
    parser.add_argument('--e_layers', type=int, default=2, help='num of encoder layers')
    parser.add_argument('--d_layers', type=int, default=1, help='num of decoder layers')
    parser.add_argument('--d_ff', type=int, default=256, help='dimension of fcn')
    parser.add_argument('--factor', type=int, default=1, help='attn factor')

    # condition_models(SVQ)
    parser.add_argument('--wFFN', type=int, default=1, help='use FFN layer')
    parser.add_argument('--svq', type=int, default=1, help='use sparse vector quantized')
    parser.add_argument('--codebook_size', type=int, default=128, help='codebook_size in sparse vector quantized')
    parser.add_argument('--sout', type=int, default=0, help='sparse linear for output')
    parser.add_argument('--individual', type=int, default=0, help='individual head; True 1 False 0')
    parser.add_argument('--revin', type=int, default=1, help='RevIN; True 1 False 0')
    parser.add_argument('--affine', type=int, default=0, help='RevIN-affine; True 1 False 0')
    parser.add_argument('--subtract_last', type=int, default=0, help='0: subtract mean; 1: subtract last')
    parser.add_argument('--length', type=int, default=96)
    parser.add_argument('--num_codebook', type=int, default=4, help='number of codebooks in sparse vector quantized')

    # condition_models(PatchTST)
    parser.add_argument('--fc_dropout', type=float, default=0.05, help='fully connected dropout')
    parser.add_argument('--head_dropout', type=float, default=0.0, help='head dropout')

    # condition_models(S-Mamba)
    parser.add_argument('--d_state', type=int, default=32, help='parameter of Mamba Block')


    # train
    parser.add_argument('--n_bins', type=int, default=10, help='')
    parser.add_argument('--PICP_range', type=lambda s: [float(x) for x in s.split(',')], default=[2.5, 97.5], help='')
    parser.add_argument('--n_blocks', type=int, default=5)
    parser.add_argument('--rmom', type=int, default=20)

    # PatchDN、NonLiear
    parser.add_argument('--patch_size', type=int, default=16, help='patch length')
    parser.add_argument('--stride', type=int, default=8, help='stride')
    parser.add_argument('--padding_patch', default='end', help='None: None; end: padding on the end')
  #  parser.add_argument('--d_model', type=int, default=1024, help='dimension of model')

    # networks
    parser.add_argument("--diff_dim", type=int, default=128, help="diffusion dimension for the model")
    parser.add_argument("--h_dim", type=int, default=128, help="hidden dimension for the model")
    parser.add_argument("--n_heads", type=int, default=4, help="number of attention heads")
    parser.add_argument('--diff_d_state', type=int, default=32)

    # optimization
    parser.add_argument("--train_flag", type=int, help="training or not", default=1)
    parser.add_argument("--train_times", type=int, default=1, help="times of training")
    parser.add_argument("--task_name", type=str, default="train", help="task name: train")
    parser.add_argument("--patience", type=int, default=15, help="early stopping patience")
    parser.add_argument("--num_epochs", type=int, default=150, help="number of epochs for train")
    parser.add_argument("--eval_frequency", type=int, default=1, help="evaluation frequency for train")
    parser.add_argument("--learning_rate", type=float, default=1e-4, help="train learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-4, help="train weight decay")
    parser.add_argument("--lr_decay", type=float, default=0.98, help="train learning rate decay")
    parser.add_argument('--loss_type', type=str, default='mse', help='loss function')

    # gpu
    parser.add_argument("--verbose", action="store_true", help="verbose", default=True)
    parser.add_argument("--use_tqdm", action="store_true", help="use tqdm", default=False)
    parser.add_argument("--use_time_features", type=bool, default=False, help="")
    parser.add_argument("--seed", type=int, default=21, help="fixed random seed")
    parser.add_argument("--device", type=str, default="cuda:1", help="device")

    _args = parser.parse_args()
    organize_file_structure(_args)

    _args.device = torch.device(_args.device)
    return _args
