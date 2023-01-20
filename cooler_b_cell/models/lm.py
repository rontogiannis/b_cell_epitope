from torch import nn
from esm.pretrained import load_model_and_alphabet

class LanguageNet(nn.Module) :
    def __init__(
        self,
        esm_model_name: str = "esm2_t30_150M_UR50D",
        reduce_dim: int = 2**30,
        finetune: bool = False,
    ) :
        super().__init__()

        model, _ = load_model_and_alphabet(esm_model_name)
        self.model = model

        if reduce_dim < self.model.embed_dim :
            self.fc = nn.Linear(self.model.embed_dim, reduce_dim)
            self.embed_dim = reduce_dim
        else :
            self.fc = nn.Identity()
            self.embed_dim = self.model.embed_dim

        for p in self.model.parameters() :
            p.requires_grad = finetune

    def forward(
        self,
        seq,
    ) :
        idx = self.model.num_layers

        seq_emb = self.model(seq, repr_layers=[idx])
        seq_emb = seq_emb["representations"][idx]

        out = self.fc(seq_emb)

        return out

