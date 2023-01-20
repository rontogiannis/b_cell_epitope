import torch
from torch import nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)

class RecurrentNet(nn.Module) :
    def __init__(
        self,
        input_size: int,
        hidden_size: int = 256,
        num_layers: int = 2,
        bias: bool = True,
        batch_first: bool = True,
        dropout: float = .0,
        bidirectional: bool = True,
    ) :
        super().__init__()

        self.model = nn.GRU(
            input_size = input_size,
            hidden_size = hidden_size,
            num_layers = num_layers,
            bias = bias,
            batch_first = batch_first,
            dropout = dropout,
            bidirectional = bidirectional,
        )

    def forward(
        self,
        emb,
        mask,
    ) :
        lens = torch.sum(mask, dim=1)

        emb_packed = pack_padded_sequence(
            emb,
            lens.cpu(),
            batch_first = True,
            enforce_sorted = False,
        )

        out, _ = self.model(emb_packed)

        out_padded, _ = pad_packed_sequence(
            out,
            batch_first = True,
            total_length = emb.shape[1],
        )

        return out_padded

