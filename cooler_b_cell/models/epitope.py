import torch
from torch import nn
from cooler_b_cell.models.lm import LanguageNet
from cooler_b_cell.models.egnn import GraphNet
from cooler_b_cell.models.rnn import RecurrentNet

class Epitope(nn.Module) :
    def __init__(
        self,
        skip_egnn: bool = False,
        skip_rnn: bool = False,
        esm_model_name: str = "esm2_t30_150M_UR50D",
        esm_reduce_dim: int = 2**30,
        esm_finetune: bool = False,
        egnn_num_layers: int = 2,
        egnn_m_dim: int = 256,
        egnn_num_nn: int = 8,
        egnn_dropout: float = .0,
        egnn_sparse_only: bool = True,
        egnn_soft_edges: bool = True,
        rnn_hidden_size: int = 256,
        rnn_num_layers: int = 2,
        rnn_bidirectional: bool = True,
        rnn_dropout: float = .0,
        mlp_hidden_dim: int = 256,
        mlp_dropout: float = .0,
    ) :
        super().__init__()

        self.lm = LanguageNet(
            esm_model_name = esm_model_name,
            reduce_dim = esm_reduce_dim,
            finetune = esm_finetune,
        )

        embed_dim = self.lm.embed_dim + 11
        mlp_embed_dim = (2 if rnn_bidirectional else 1)*rnn_hidden_size if not skip_rnn else embed_dim

        self.egnn = GraphNet(
            num_layers = egnn_num_layers,
            dim = embed_dim,
            edge_dim = 48,
            m_dim = egnn_m_dim,
            fourier_features = 0,
            num_nearest_neighbors = egnn_num_nn,
            dropout = egnn_dropout,
            init_eps = 1e-3,
            norm_feats = False,
            norm_coors = True,
            norm_coors_scale_init = 1e-2,
            update_feats = True,
            update_coors = False,
            only_sparse_neighbors = egnn_sparse_only,
            valid_radius = float("inf"),
            m_pool_method = "sum",
            soft_edges = egnn_soft_edges,
            coor_weights_clamp_value = None,
        ) if not skip_egnn else nn.Identity()

        self.rnn = RecurrentNet(
            input_size = embed_dim,
            hidden_size = rnn_hidden_size,
            num_layers = rnn_num_layers,
            bias = True,
            batch_first = True,
            dropout = rnn_dropout,
            bidirectional = rnn_bidirectional,
        ) if not skip_rnn else nn.Identity()

        self.mlp = nn.Sequential(
            nn.Linear(mlp_embed_dim, mlp_hidden_dim),
            nn.ReLU(),
            nn.Dropout(mlp_dropout),
            nn.Linear(mlp_hidden_dim, 3),
        )

        self.skip_egnn = skip_egnn
        self.skip_rnn = skip_rnn

    def forward(
        self,
        tokens,
        coord,
        node_feat,
        edge_feat,
        mask,
        graph,
    ) :
        emb = self.lm(tokens)[:,1:,:]
        emb = torch.cat((emb, node_feat), 2)
        emb = emb if self.skip_egnn else self.egnn(emb, coord, edge_feat, mask, graph)
        emb = emb if self.skip_rnn else self.rnn(emb, mask)
        out = self.mlp(emb)
        return out

