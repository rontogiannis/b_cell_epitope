import torch
from torch import nn
from egnn_pytorch import EGNN

class GraphUnit(nn.Module) :
    def __init__(
        self,
        dim,
        edge_dim: int = 0,
        m_dim: int = 16,
        fourier_features: int = 0,
        num_nearest_neighbors: int = 0,
        dropout: float = 0.0,
        init_eps: float = 1e-3,
        norm_feats: bool = False,
        norm_coors: bool = True,
        norm_coors_scale_init: float = 1e-2,
        update_feats: bool = True,
        update_coors: bool = False,
        only_sparse_neighbors: bool = True,
        valid_radius: float = .0,
        m_pool_method: str = "sum",
        soft_edges: bool = True,
        coor_weights_clamp_value: float = None,
    ) :
        super().__init__()

        self.model = EGNN(
            dim,
            edge_dim,
            m_dim,
            fourier_features,
            num_nearest_neighbors,
            dropout,
            init_eps,
            norm_feats,
            norm_coors,
            norm_coors_scale_init,
            update_feats,
            update_coors,
            only_sparse_neighbors,
            valid_radius,
            m_pool_method,
            soft_edges,
            coor_weights_clamp_value,
        )

    def forward(
        self,
        params,
    ) :
        emb, coord, edge_feat, mask, graph = params

        out, coord_new = self.model(
            emb,
            coord,
            edges = edge_feat,
            mask = mask,
            adj_mat = graph,
        )

        return out, coord_new, edge_feat, mask, graph

class GraphNet(nn.Module) :
    def __init__(
        self,
        num_layers: int,
        dim: int,
        edge_dim: int = 0,
        m_dim: int = 16,
        fourier_features: int = 0,
        num_nearest_neighbors: int = 0,
        dropout: float = 0.0,
        init_eps: float = 1e-3,
        norm_feats: bool = False,
        norm_coors: bool = True,
        norm_coors_scale_init: float = 1e-2,
        update_feats: bool = True,
        update_coors: bool = False,
        only_sparse_neighbors: bool = True,
        valid_radius: float = .0,
        m_pool_method: str = 'sum',
        soft_edges: bool = True,
        coor_weights_clamp_value: float = None,
    ) :
        super().__init__()

        self.model = nn.Sequential(*[GraphUnit(
            dim,
            edge_dim,
            m_dim,
            fourier_features,
            num_nearest_neighbors,
            dropout,
            init_eps,
            norm_feats,
            norm_coors,
            norm_coors_scale_init,
            update_feats,
            update_coors,
            only_sparse_neighbors,
            valid_radius,
            m_pool_method,
            soft_edges,
            coor_weights_clamp_value
        ) for i in range(num_layers)])

    def forward(
        self,
        emb,
        coord,
        edge_feat,
        mask,
        graph,
    ) :
        out_tuple = self.model((emb, coord, edge_feat, mask, graph))

        return out_tuple[0]

class GatedGraphNet(nn.Module) :
    def __init__(
        self,
        num_layers: int,
        dim: int,
        edge_dim: int = 0,
        m_dim: int = 16,
        fourier_features: int = 0,
        num_nearest_neighbors: int = 0,
        dropout: float = 0.0,
        init_eps: float = 1e-3,
        norm_feats: bool = False,
        norm_coors: bool = True,
        norm_coors_scale_init: float = 1e-2,
        update_feats: bool = True,
        update_coors: bool = False,
        only_sparse_neighbors: bool = True,
        valid_radius: float = .0,
        m_pool_method: str = 'sum',
        soft_edges: bool = True,
        coor_weights_clamp_value: float = None,
    ) :
        super().__init__()

        self.shallow = nn.Sequential(*[GraphUnit(
            dim,
            edge_dim,
            m_dim,
            fourier_features,
            num_nearest_neighbors,
            dropout,
            init_eps,
            norm_feats,
            norm_coors,
            norm_coors_scale_init,
            update_feats,
            update_coors,
            only_sparse_neighbors,
            valid_radius,
            m_pool_method,
            soft_edges,
            coor_weights_clamp_value
        ) for i in range(num_layers-1)])

        self.deep = GraphUnit(
            dim,
            edge_dim,
            m_dim,
            fourier_features,
            num_nearest_neighbors,
            dropout,
            init_eps,
            norm_feats,
            norm_coors,
            norm_coors_scale_init,
            update_feats,
            update_coors,
            only_sparse_neighbors,
            valid_radius,
            m_pool_method,
            soft_edges,
            coor_weights_clamp_value
        )

        self.gate = nn.Linear(2*dim, dim)

    def forward(
        self,
        emb,
        coord,
        edge_feat,
        mask,
        graph,
    ) :
        shallow_emb = self.shallow((emb, coord, edge_feat, mask, graph))[0]
        deep_emb = self.deep((shallow_emb, coord, edge_feat, mask, graph))[0]
        emb = torch.cat((shallow_emb, deep_emb), 2)
        emb = self.gate(emb)

        return emb

