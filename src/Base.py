import os
import logging
import datetime
import torch
import Config, Datasets, Model, Trainer, Utils
from transformers import AutoTokenizer, BertTokenizerFast
import pandas as pd
import time
import gc
from tqdm import tqdm
import copy
import numpy as np
import json
logging.basicConfig(format='%(asctime)s - %(levelname)s - %(message)s', datefmt='%Y-%m-%d %H:%M:%S')


@Utils.timer
def XXXBase(args):
    # tokenizer
    # logging.debug(args.tokenizer_path)
    # tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    ############
    # training sample
    # train_sample = Datasets.prepare_training_samples()
    ############
    # datasets
    # train_datasets = Datasets.BaseDatasets()
    # valid_datasets = Datasets.BaseDatasets()
    # train_iter = torch.utils.data.DataLoader(train_datasets, batch_size=args.train_batch_size, shuffle=True)
    # logging.info(f"Train Size: {len(train_iter)}")
    ############
    # trainer
    # trainer = Trainer.BaseTrainer(args)
    # trainer.trainer_init(len(train_iter), valid_datasetes)
    ############    
    for epoch in range(args.epoch):
        logging.info(f"Start Epoch: {epoch}")
        t_s = time.time()
        loss = 0
        if args.debug:
            for batch in tqdm(train_iter):
                loss += trainer.step(batch)
        else:
            for batch in train_iter:
                loss += trainer.step(batch)
        logging.info("Train Loss: {:.4f}".format(loss / len(train_iter)))
        t_e = time.time()
        logging.info("Cost {:.2f} s.".format(t_e - t_s))
    score_maxn = trainer.score_maxn
    logging.info("Best Score: {:.6f}".format(score_maxn))
    del train_samples, valid_samples
    del train_datasets, valid_datasets
    del train_iter, trainer
    gc.collect()
    return score_maxn


@Utils.timer
def main(args):
    if args.mode not in ["base"]:
        raise
    model_save = "/".join([args.model_save, Utils.d2s(datetime.datetime.now(), time=True)])
    if not args.debug:
        if os.path.exists(model_save):
            logging.warning("save path exists, sleep 60s")
            raise
        else:
            os.mkdir(model_save)
            args.model_save = model_save
    MODEL_PREFIX = args.model_save
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    logging.info(f"device: {args.device}")
    if args.train_all:
        num = args.fold
        for fold in range(num):
            args.fold = fold
            args.model_save = "/".join([MODEL_PREFIX, f"Fold_{args.fold}.bin"])
            logging.info(f"model save path: {args.model_save}")
            if args.mode == "base":
                XXXBase(args)
    else:
        args.model_save = "/".join([MODEL_PREFIX, f"Fold_{args.fold}.bin"])
        logging.info(f"model save path: {args.model_save}")
        if args.mode == "base":
            XXXBase(args)


@Utils.timer
def predict(args):
    args.device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    raise

if __name__ == "__main__":
    args = Config.BaseConfig()
    Utils.set_seed(args.seed)
    if not args.debug:
        logging.getLogger().setLevel(logging.INFO)
    else:
        logging.getLogger().setLevel(logging.DEBUG)
    if args.train:
        logging.info(f"args: {args}".replace(" ", "\n"))
        main(args)
    if args.predict:
        if args.train:
            args.model_load = args.model_save
        predict(args)
