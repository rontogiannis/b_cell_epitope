import torch
from torch import nn
from torch.utils.data import Dataset
from esm.pretrained import load_model_and_alphabet
from cooler_b_cell.scripts.utils import load_jsonl

class Epitope(Dataset) :
    def __init__(
        self,
        path: str,
        esm_model_name: str = "esm2_t30_150M_UR50D",
        padded_length: int = 1024,
    ) :
        super().__init__()

        _, tokenizer = load_model_and_alphabet(esm_model_name)
        self.tokenizer = tokenizer
        self.padded_length = padded_length

        raw = load_jsonl(path)
        tokens, coord, node_feat, edge_feat, mask, graph, y = self._setup(raw)

        self.tokens = tokens # torch.tensor(tokens, dtype=torch.long)
        self.coord = torch.tensor(coord, dtype=torch.float)
        self.node_feat = torch.tensor(node_feat, dtype=torch.float)
        self.edge_feat = edge_feat
        self.mask = torch.tensor(mask, dtype=torch.long).bool()
        self.graph = graph
        self.y = torch.tensor(y, dtype=torch.long)

        self.graph_dim = (padded_length+2, padded_length+2)
        self.edge_feat_dim = (padded_length+2, padded_length+2, 48)

    def _pad(self, whatever, empty) :
        return whatever + (self.padded_length-len(whatever)+1)*[empty]

    def _setup(self, raw) :
        pids = []
        seqs = []
        tokens = []
        coord = []
        node_feat = []
        edge_feat = []
        mask = []
        graph = []
        y = []

        for line in raw :
            pid_i = line["pdb"]
            seq_i = line["seq"]
            coord_i = line["coord"]
            node_feat_i = line["node_feat"]
            edge_feat_i = line["edge_feat"]
            graph_i = line["graph"]
            y_i = line["label"]

            pids.append(pid_i)
            seqs.append(seq_i)
            coord.append(self._pad(coord_i, [.0]*3))
            node_feat.append(self._pad(node_feat_i, [.0]*11))
            edge_feat.append(edge_feat_i)
            mask.append(self._pad([1]*len(seq_i), 0))
            graph.append(graph_i)
            y.append(self._pad(y_i, .0))

        seq_with_id = list(zip(pids, seqs)) + [("dummy", "<mask>"*self.padded_length)]
        tokens = self.tokenizer.get_batch_converter()(seq_with_id)[2][:-1]

        return tokens, coord, node_feat, edge_feat, mask, graph, y

    def __len__(self) :
        return len(self.tokens)

    def __getitem__(self, idx) :
        graph_tensor = torch.tensor(self.graph[idx], dtype=torch.long)
        edge_feat_tensor = torch.tensor(self.edge_feat[idx], dtype=torch.long)

        graph_dense = torch.sparse.FloatTensor(graph_tensor, torch.ones(graph_tensor.shape[1]), self.graph_dim).to_dense().bool()
        edge_feat_dense = torch.sparse.FloatTensor(edge_feat_tensor, torch.ones(edge_feat_tensor.shape[1]), self.edge_feat_dim).to_dense()

        return {
            "tokens": self.tokens[idx],
            "coord": self.coord[idx],
            "node_feat": self.node_feat[idx],
            "edge_feat": edge_feat_dense[1:,1:,:],
            "mask": self.mask[idx],
            "graph": graph_dense[1:,1:],
            "y": self.y[idx],
        }

