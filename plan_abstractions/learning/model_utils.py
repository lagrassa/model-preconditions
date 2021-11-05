from collections import OrderedDict
import torch as nn


def make_mlp(in_size, layer_sizes, act='tanh', last_act=True, dropout=0, prefix=''):
    if act =='tanh':
        act_f = nn.Tanh()
    elif act in ('relu', 'ReLU'):
        act_f = nn.ReLU(inplace=True)
    elif act in ('leakyrelu', 'LeakyReLU'):
        act_f = nn.LeakyReLU(inplace=True)
    elif act == 'sigmoid':
        act_f = nn.Sigmoid()
    else:
        raise ValueError(f'Unknown act: {act}')

    layers = []
    for i, layer_size in enumerate(layer_sizes):
        layers.append((f'{prefix}_linear{i}', nn.Linear(in_size, layer_size)))
        if i < len(layer_sizes) - 1:
            if dropout > 0:
                layers.append((f'{prefix}_dropout{i}', nn.Dropout(dropout)))
            layers.append((f'{prefix}_{act}{i}', act_f))
        else:
            if last_act:
                layers.append((f'{prefix}_{act}{i}', act_f))
        in_size = layer_size
    return nn.Sequential(OrderedDict(layers))