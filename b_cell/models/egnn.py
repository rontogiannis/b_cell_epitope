from torch import nn
from egnn_pytorch import EGNN

class EGNNModule(nn.Module) :
    def __init__(
        self,
        embedding_dim: int,
        dropout: float,
        egnn_dim: int,
        egnn_nn: int,
        egnn_edge_dim: int,
        egnn_valid_radius: float,
        only_sparse: bool,
    ) :
        super().__init__()

        self.egnn = EGNN(
            dim=embedding_dim,
            edge_dim=egnn_edge_dim,
            m_dim=egnn_dim,
            fourier_features=0,
            num_nearest_neighbors=egnn_nn,
            dropout=dropout,
            norm_feats=False,
            norm_coors=True,
            update_feats=True,
            update_coors=False,
            only_sparse_neighbors=only_sparse,
            valid_radius=egnn_valid_radius,
            m_pool_method="sum",
            soft_edges=False,
            coor_weights_clamp_value=None,
        )

    def forward(self, params) :
        emb, coors, adj, feat, mask = params
        out, coors_upd = self.egnn(emb, coors, edges=feat, mask=mask, adj_mat=adj)
        return out, coors_upd, adj, feat, mask
