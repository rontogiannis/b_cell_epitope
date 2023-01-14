from torch import nn
from egnn_pytorch import EGNN

required = {
    "dim",
    "edge_dim",
}

default_config = {
    "m_dim": 256,
    "fourier_features": 0,
    "num_nearest_neighbors": 0,
    "dropout": .0,
    "norm_feats": False,
    "norm_coors": True,
    "update_feats": True,
    "update_coors": False,
    "only_sparse_neighbors": True,
    "valid_radius": .0,
    "m_pool_method": "sum",
    "soft_edges": True,
    "coor_weights_clamp_value": None,
}

class GraphUnit(nn.Module) :
    def __init__(self, **config) :
        assert required.issubset(config)

        super().__init__()

        self.h = { **default_config, **config }
        self.model = EGNN(**self.h)

    def forward(self, **params) :
        out, crd = self.model(
            params["embeddings"],
            params["coordinates"],
            edges = params["edge features"],
            mask = params["mask"],
            adj_mat = params["graph"],
        )

        return {
            "embeddings": out,
            "coordinates": crd,
            "edge_features": params["edge features"],
            "mask": params["mask"],
            "graph": params["graph"],
        }

class GraphNet(nn.Module) :
    def __init__(self, num_layers, **config) :
        assert required.issubset(config)

        super().__init__()

        self.num_layers = num_layers
        self.h = { **default_config, **config }
        self.model = nn.Sequential(*[GraphUnit(**self.h) for i in range(num_layers)])

    def forward(self, **params) :
        return self.model(**params)["embeddings"]

