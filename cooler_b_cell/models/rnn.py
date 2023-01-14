from torch import nn
from torch.nn.utils.rnn import (
    pack_padded_sequence,
    pad_packed_sequence,
)

required = {
    "input_size",
}

default_config = {
    "hidden_size": 256,
    "num_layers": 2,
    "batch_first": True,
    "bias": True,
    "dropout": .0,
    "bidirectional": True,
}

class RecurrentNet(nn.Module) :
    def __init__(self, **config) :
        assert required.issubset(config)

        super().__init__()

        self.h = { **default_config, **config }
        self.model = nn.GRU(**self.h)

    def forward(self, **params) :
        emb_packed = pack_padded_sequence(
            params["embeddings"],
            params["lengths"].cpu(),
            batch_first = True,
            enforce_sorted = False,
        )

        out, hn = self.model(emb_packed)

        out_padded, _ = pad_packed_sequence(
            out,
            batch_first = True,
            total_length = params["embeddings"].shape[1],
        )

        return {
            "embeddings": out_padded,
            "hidden_state": hn,
        }

