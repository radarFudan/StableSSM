import torch
import torch.nn as nn
import torch.nn.functional as F


class LSTM(nn.Module):
    def __init__(
        self,
        rec1_size: int = 128,
        activation: str = "linear",
        return_sequences: bool = True,
        training: bool = True,  # currently useless
    ):
        super().__init__()

        self.hidden_size = rec1_size
        self.num_layers = 1
        self.layer = torch.nn.LSTM(
            input_size=rec1_size,
            hidden_size=self.hidden_size,
            num_layers=self.num_layers,
            bias=True,
            batch_first=True,
            dropout=0.0,
            bidirectional=False,
            dtype=torch.float32,
        )

        if activation == "linear":
            self.activation = torch.nn.Identity()
        elif activation == "tanh":
            self.activation = torch.tanh
        elif activation == "hardtanh":
            self.activation = torch.nn.functional.hardtanh

    def forward(self, x, h0=None, c0=None):
        B, _, _ = x.shape  # B * T * D
        h0 = (
            torch.randn(self.num_layers, B, self.hidden_size, dtype=torch.float32).to(x.device)
            if h0 is None
            else h0
        )
        c0 = (
            torch.randn(self.num_layers, B, self.hidden_size, dtype=torch.float32).to(x.device)
            if c0 is None
            else c0
        )
        output, (hn, cn) = self.layer(x, (h0, c0))

        return output

    def stability_margin(self):
        """Return the stability margin of the model."""
        raise NotImplementedError

    def perturb_weight_initialization(self):
        """Perturb the weight initialization to make the model unstable."""
        raise NotImplementedError


if __name__ == "__main__":
    pass
