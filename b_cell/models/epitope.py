import esm
import torch

from torch import nn
from b_cell.models.egnn import EGNNModule
from b_cell.models.rnn import RNNModule
from b_cell.models.minifold import MiniFold

class EpitopePredictionModel(nn.Module) :
    def __init__(
        self,
        esm_model_name: str,
        esm_layer_cnt: int,
        use_egnn: bool,
        use_rho: bool,
        use_iedb: bool,
        use_dssp: bool,
        use_rnn: bool,
        esm_dim: int,
        egnn_dim: int,
        egnn_edge_dim: int,
        egnn_nn: int,
        egnn_layers: int,
        mlp_hidden_dim: int,
        dropout: float,
        finetune_lm: bool = False,
        total_padded_length: int = 950,
        rnn_hidden_dim: int = 512,
        rnn_num_layers: int = 2,
        rnn_bidirectional: bool = True,
        finetune_mlp_only: bool = False,
        use_minifold: bool = False,
        finetune_minifold: bool = False,
    ) :
        super().__init__()

        # Load ESM language model
        try :
            esm_model, alphabet = torch.hub.load("facebookresearch/esm:main", esm_model_name)
        except Exception :
            bar = getattr(esm.pretrained, esm_model_name)
            esm_model, alphabet = bar()

        self.esm_model = esm_model
        self.alphabet = alphabet
        self.ll_idx = esm_layer_cnt

        for param in self.esm_model.parameters() :
            param.requires_grad = finetune_lm and (not finetune_mlp_only)

        if use_minifold :
            self.minifold = MiniFold.load_from_checkpoint("/Mounts/rbg-storage1/users/jwohlwend/minifold_v1.ckpt")

            for p in self.minifold.parameters() :
                p.requires_grad = finetune_minifold

        # +2  for the IEDB embeddings
        # +5  for rho TODO make len(lambda) customizable
        # +11 for the dssp features
        embedding_dim = (esm_dim if not use_minifold else self.minifold.lm.embed_dim) + \
            (2 if use_iedb else 0) + \
            (5 if use_rho else 0) + \
            (11 if use_dssp else 0)

        d = 2 if rnn_bidirectional else 1

        # flags
        self.finetune_lm = finetune_lm
        self.finetune_mlp_only = finetune_mlp_only
        self.use_egnn = use_egnn
        self.use_rho = use_rho
        self.use_iedb = use_iedb
        self.use_dssp = use_dssp
        self.use_rnn = use_rnn
        self.use_minifold = use_minifold

        # Equivariant GNN
        self.egnn = nn.Sequential(
            *[EGNNModule(
                embedding_dim=embedding_dim,
                dropout=0, # TODO maybe play around with this
                egnn_dim=egnn_dim,
                egnn_nn=egnn_nn,
                egnn_edge_dim=egnn_edge_dim,
                egnn_valid_radius=0,
                only_sparse=True,
            ) for layer in range(egnn_layers)]
        ) if use_egnn else None

        # RNN of choice (bidirectional GRU in this case)
        self.rnn = RNNModule(
            embedding_dim=embedding_dim,
            total_padded_length=total_padded_length,
            dropout=dropout,
            rnn_hidden_dim=rnn_hidden_dim,
            rnn_num_layers=rnn_num_layers,
            rnn_bidirectional=rnn_bidirectional,
            return_hidden_state=False,
        ) if use_rnn else None

        # multi-layer perceptron
        self.mlp = nn.Sequential(
            nn.Linear(embedding_dim if not use_rnn else d*rnn_hidden_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, mlp_hidden_dim//2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim//2, 1),
            nn.Sigmoid(),
        )

        if finetune_mlp_only :
            for p in self.parameters() :
                p.requires_grad = False
            for p in self.mlp.parameters() :
                p.requires_grad = True

    def forward(self, params, mask) :
        X, lens, coors, rho, adj, feat, dssp_feat, iedb_emb = params

        # language model embeddings
        if not self.use_minifold :
            dct = self.esm_model(X, repr_layers=[self.ll_idx], return_contacts=False)
            emb = dct["representations"][self.ll_idx]
        else :
            _, emb = self.minifold(X)

        # concatenate embeddings
        emb = torch.cat((emb, dssp_feat), 2) if self.use_dssp else emb
        emb = torch.cat((emb, iedb_emb), 2) if self.use_iedb else emb
        emb = torch.cat((emb, rho), 2) if self.use_rho else emb

        # apply EGNN
        emb = self.egnn((emb, coors, adj, feat, mask))[0] if self.use_egnn else emb

        # apply RNN
        emb = self.rnn(emb, lens) if self.use_rnn else emb

        # apply MLP
        out = self.mlp(emb)

        return out

