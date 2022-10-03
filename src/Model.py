import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoConfig
from transformers.models.bert.modeling_bert import BertPreTrainedModel, BertEmbeddings, BertEncoder, BertOnlyMLMHead
import logging
import math


class ModelConfig(object):
    def __init__(self, args):
        self.pretrain_path = args.pretrain_path
        self.hidden_dropout_prob = 0.1
        self.layer_norm_eps = 1e-7
        self.device = args.device
        self.dropout = args.dropout


class Swish(nn.Module):
    def __init__(self, inplace=True):
        super(Swish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        if self.inplace:
            x.mul_(torch.sigmoid(x))
            return x
        else:
            return x * torch.sigmoid(x)


class Activation(nn.Module):
    def __init__(self, name="swish"):
        super(Activation, self).__init__()
        if name not in ["swish", "relu", "gelu"]:
            raise
        if name == "swish":
            self.net = Swish()
        elif name == "relu":
            self.net = nn.ReLU()
        elif name == "gelu":
            self.net = nn.GELU()
    
    def forward(self, x):
        return self.net(x)


class Dence(nn.Module):
    def __init__(self, i_dim, o_dim, activation="swish"):
        super(Dence, self).__init__()
        self.dence = nn.Sequential(
            nn.Linear(i_dim, o_dim),
            # nn.ReLU(),
            Activation(activation),
        )

    def forward(self, x):
        return self.dence(x)


class BatchDence(nn.Module):
    def __init__(self, i_dim, o_dim, activation="swish"):
        super(BatchDence, self).__init__()
        self.dence = nn.Sequential(
            nn.Linear(i_dim, o_dim),
            nn.BatchNorm1d(o_dim),
            # nn.ReLU(),
            Activation(activation),
        )

    def forward(self, x):
        return self.dence(x)


class MeanPooling(nn.Module):
    def __init__(self):
        super(MeanPooling, self).__init__()
    
    def forward(self, last_hidden_state, attention_mask):
        input_mask_extended = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
        sum_embeddings = torch.sum(last_hidden_state * input_mask_extended, 1)
        sum_mask = input_mask_extended.sum(1)
        sum_mask = torch.clamp(sum_mask, min=1e-9)
        mean_embeddings = sum_embeddings / sum_mask
        return mean_embeddings