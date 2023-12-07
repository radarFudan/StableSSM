import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat

from mamba_ssm import Mamba


class MambaModel(nn.Module):
    def __init__(
        self,
        rec1_size=256,
        n_layers=4,
        dropout=0.2,
        dt=0.33,
        prenorm=False,
        parameterization="exp",  # this is a kernel_arg
        return_seq: bool =False,
    ):
        super().__init__()

        self.prenorm = prenorm
        self.return_seq = return_seq

        # Stack Mamba layers as residual blocks
        self.mamba_layers = nn.ModuleList()
        self.norms = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        for _ in range(n_layers):
            self.mamba_layers.append(
                Mamba(
                    rec1_size,
                    dropout=dropout,
                    transposed=True,
                    lr=min(0.001, dt),
                    parameterization=parameterization,
                )
            )
            self.norms.append(nn.LayerNorm(rec1_size))
            self.dropouts.append(nn.Dropout1d(dropout))

    def forward(self, x):
        """Input x is shape (B, L, d_model)"""

        x = x.transpose(-1, -2)  # (B, L, d_model) -> (B, d_model, L)
        for layer, norm, dropout in zip(self.stablessm_layers, self.norms, self.dropouts):
            # Each iteration of this loop will map (B, d_model, L) -> (B, d_model, L)

            z = x
            if self.prenorm:
                # Prenorm
                z = norm(z.transpose(-1, -2)).transpose(-1, -2)

            # Apply StableSSM block: we ignore the state input and output
            z, _ = layer(z)

            # Dropout on the output of the StableSSM block
            z = dropout(z)

            # Residual connection
            x = z + x

            if not self.prenorm:
                # Postnorm
                x = norm(x.transpose(-1, -2)).transpose(-1, -2)

        x = x.transpose(-1, -2)  # (B, d_model, L) -> (B, L, d_model)

        # Pooling: average pooling over the sequence length
        if not self.return_seq:
            x = x.mean(dim=1)  # This is actually a linear convolution layer...
        else:
            x = x

        return x
