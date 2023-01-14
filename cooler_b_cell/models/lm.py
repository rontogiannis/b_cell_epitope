from torch import nn, randint
from esm.pretrained import load_model_and_alphabet

required = set()

default_config = {
    "esm_model_name": "esm2_t30_150M_UR50D",
    "reduce_dim": 2**30,
    "finetune": False,
}

class LanguageNet(nn.Module) :
    def __init__(self, **config) :
        assert required.issubset(config)

        super().__init__()

        self.h = { **default_config, **config }

        model, alphabet = load_model_and_alphabet(self.h["esm_model_name"])

        self.model = model
        self.alphabet = alphabet

        self.fc = nn.Linear(self.model.embed_dim, self.h["reduce_dim"]) if self.h["reduce_dim"] < self.model.embed_dim else nn.Identity()

        for p in self.model.parameters() :
            p.requires_grad = self.h["finetune"]

    def forward(self, **params) :
        idx = self.model.num_layers

        seq_emb = self.model(params["seq"], repr_layers=[idx])
        seq_emb = seq_emb["representations"][idx]

        out = self.fc(seq_emb)

        return out

