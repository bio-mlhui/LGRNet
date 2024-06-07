
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat, rearrange, reduce
from typing import Any, Optional
from torch import Tensor
from .utils import _get_activation_fn

class MLP(nn.Module):
    """ Very simple multi-layer perceptron (also called FFN)"""
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super().__init__()
        self.num_layers = num_layers
        h = [hidden_dim] * (num_layers - 1)
        self.layers = nn.ModuleList(nn.Linear(n, k) for n, k in zip([input_dim] + h, h + [output_dim]))

    def forward(self, x):
        for i, layer in enumerate(self.layers):
            x = F.relu(layer(x)) if i < self.num_layers - 1 else layer(x)
        return x

class Linear_NormAct(nn.Linear):
    def __init__(self, *args, **kwargs):
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)
        out_features = kwargs['out_features']
        if norm == None:
            self.norm = None
        elif norm == 'ln':
            self.norm = nn.LayerNorm(out_features)
        else:
            raise ValueError()
        
        self.activation = _get_activation_fn(activation)

    def forward(self, x):
        x = F.linear(x, self.weight, self.bias)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class Conv2d_NormAct(torch.nn.Conv2d):
    def __init__(self, *args, **kwargs):
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)
        out_dim = kwargs['out_channels']
        if norm is None:
            self.norm = None
        elif norm == 'bn2d':
            # b c h w
            self.norm = nn.BatchNorm2d(out_dim)
        elif 'gn' in norm:
            # b c ..
            n_groups = int(norm.split('_')[-1])
            self.norm = nn.GroupNorm(n_groups, out_dim)
        else:
            raise ValueError()
        self.activation = _get_activation_fn(activation)

    def forward(self, x):
        # b c h w
        x = self._conv_forward(x, self.weight, self.bias)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

class Conv3d_NormAct(torch.nn.Conv3d):
    def __init__(self, *args, **kwargs):
        norm = kwargs.pop("norm", None)
        activation = kwargs.pop("activation", None)
        super().__init__(*args, **kwargs)
        out_dim = kwargs['out_channels']
        if norm == None:
            self.norm = None
        elif 'gn' in norm:
            n_groups = int(norm.split('_')[-1])
            self.norm = nn.GroupNorm(n_groups, out_dim)
        else:
            raise ValueError()
        self.activation = _get_activation_fn(activation)

    def forward(self, x):
        x = self._conv_forward(x, self.weight, self.bias)
        if self.norm is not None:
            x = self.norm(x)
        if self.activation is not None:
            x = self.activation(x)
        return x

