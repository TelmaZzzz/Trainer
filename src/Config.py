import argparse


def BaseConfig(parser):
    parser.add_argument("--train", action="store_true")
    parser.add_argument("--predict", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--data_path", type=str)
    parser.add_argument("--train_path", type=str)
    parser.add_argument("--valid_path", type=str)
    parser.add_argument("--model_save", type=str)
    parser.add_argument("--model_load", type=str)
    parser.add_argument("--output_path", type=str)
    parser.add_argument("--seed", type=int, default=959)
    parser.add_argument("--fold", type=int, default=1)
    parser.add_argument("--train_batch_size", type=int, default=16)
    parser.add_argument("--valid_batch_size", type=int, default=32)
    parser.add_argument("--scheduler", type=str, default="CosineAnnealingLR")
    parser.add_argument("--model_name", type=str, default="v1")
    parser.add_argument("--mode", type=str, default="base")
    # Trainer Config
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--min_lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--epoch", type=int, default=10)
    parser.add_argument("--opt_step", type=int, default=1)
    parser.add_argument("--eval_step", type=int, default=500)
    parser.add_argument("--Tmax", type=int, default=500)
    parser.add_argument("--max_norm", type=int, default=1)
    parser.add_argument("--warmup_step", type=int, default=-1)
    parser.add_argument("--warmup_rate", type=float, default=0.1)
    # Model Config
    parser.add_argument("--pretrain_path", type=str)
    parser.add_argument("--tokenizer_path", type=str)
    parser.add_argument("--dropout", type=float, default=0.1)
    # Other Config
    parser.add_argument("--fix_length", type=int, default=512)
    parser.add_argument("--ema", action="store_true")
    parser.add_argument("--mask_prob", type=float, default=0.0)
    parser.add_argument("--mask_ratio", type=float, default=0.0)
    parser.add_argument("--awp", action="store_true")
    parser.add_argument("--awp_up", type=float, default=-1)
    parser.add_argument("--awp_lr", type=float, default=1e-6)
    parser.add_argument("--awp_eps", type=float, default=1e-6)
    parser.add_argument("--fgm_up", type=float, default=-1)
    parser.add_argument("--fgm", action="store_true")
    parser.add_argument("--fp16", action="store_true")
    parser.add_argument("--pgd", action="store_true")
    parser.add_argument("--pgd_k", type=int, default=3)
    parser.add_argument("--swa", action="store_true")
    parser.add_argument("--swa_start_step", type=int, default=100)
    parser.add_argument("--swa_update_step", type=int, default=10)
    parser.add_argument("--swa_lr", type=float, default=1e-4)
    parser.add_argument("--rdrop", action="store_true")
    parser.add_argument("--rdrop_alpha", type=float, default=1.0)
    parser.add_argument("--da", action="store_true")
    parser.add_argument("--da_path", type=str)
    parser.add_argument("--train_all", action="store_true")
    parser.add_argument("--patience_maxn", type=int, default=10)
    return parser


def GenerateConfig(parser):
    parser.add_argument("--max_length", type=int, default=32)
    parser.add_argument("--min_length", type=int, default=0)
    parser.add_argument("--do_sample", action="store_true")
    parser.add_argument("--early_stop", action="store_true")
    parser.add_argument("--num_beams", type=int, default=0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--top_k", type=int, default=15)
    parser.add_arugment("--top_p", type=float, default=1.0)
    return parser


def Config():
    parser = argparse.ArgumentParser()
    parser = BaseConfig(parser)
    parser = GenerateConfig(parser)
    args = parser.parse_args()
    return args