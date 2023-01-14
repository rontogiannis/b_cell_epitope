import pytorch_lightning as pl
from torch import nn
from esm.pretrained import load_model_and_alphabet

class MiniFold(pl.LightningModule):
    def __init__(
        self,
        esm_model_name="esm2_t12_35M_UR50D",
        lr: float = 1e-4,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.lr = lr
        self.esm_model_name = esm_model_name

        # Language model encoder
        lm, _ = load_model_and_alphabet(esm_model_name)
        self.lm = lm
        self.lm.lm_head = nn.Identity()
        self.lm.contact_head = nn.Identity()

        # Remove some layers
        self.fc = nn.Linear(self.lm.embed_dim, 3)

    def forward(self, X):
        # Compute residue embeddings
        idx = self.lm.num_layers
        seq_emb = self.lm(X, repr_layers=[idx])
        seq_emb = seq_emb["representations"][idx]

        # Compute prediction
        preds = self.fc(seq_emb)
        return preds, seq_emb
