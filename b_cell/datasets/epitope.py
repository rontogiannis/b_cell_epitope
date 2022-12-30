import torch

from torch.utils.data import Dataset
from b_cell.scripts.utils import build_graph, residue_depth, get_dssp_feats, get_coords, load_fasta, tokenize, pad

class EpitopeDataset(Dataset) :
    def __init__(
        self,
        tokenizer,
        fasta: str,
        pdb_dir: str,
        coord_dir: str,
        dssp_dir: str,
        graph_dir: str,
        rho_dir: str,
        is_iedb: bool,
        max_pad: int,
        d_seq: int,
        k: int,
        radius: float,
    ) :
        X, lengths, mask, coord, rho, adj, feat, dssp_feats, iedb_emb, y, max_pad, d_seq = self._setup(
            tokenizer,
            fasta,
            pdb_dir,
            coord_dir,
            dssp_dir,
            graph_dir,
            rho_dir,
            1 if is_iedb else 0,
            max_pad,
            d_seq,
            k,
            radius,
        )

        self.X = torch.tensor(X, dtype=torch.long)
        self.lengths = torch.tensor(lengths, dtype=torch.long)
        self.mask = torch.tensor(mask, dtype=torch.long).bool()
        self.coord = torch.tensor(coord, dtype=torch.float)
        self.rho = torch.tensor(rho, dtype=torch.float)
        self.adj = adj
        self.feat = feat
        self.dssp_feats = torch.tensor(dssp_feats, dtype=torch.float)
        self.iedb_emb = torch.tensor(iedb_emb, dtype=torch.float)
        self.y = torch.tensor(y, dtype=torch.long)

        self.adj_dim = (max_pad+2, max_pad+2)
        self.feat_dim = (max_pad+2, max_pad+2, 21+21+2*d_seq+2)

    def __len__(self) :
        return len(self.X)

    def __getitem__(self, idx) :
        adj_tensor = torch.tensor(self.adj[idx], dtype=torch.long)
        feat_tensor = torch.tensor(self.feat[idx], dtype=torch.long)

        adj_dense = torch.sparse.FloatTensor(adj_tensor, torch.ones(adj_tensor.shape[1]), self.adj_dim).to_dense().bool()
        feat_dense = torch.sparse.FloatTensor(feat_tensor, torch.ones(feat_tensor.shape[1]), self.feat_dim).to_dense()

        return (
            self.X[idx],
            self.lengths[idx],
            self.coord[idx],
            self.rho[idx],
            adj_dense,
            feat_dense,
            self.dssp_feats[idx],
            self.iedb_emb[idx]
        ), self.mask[idx], self.y[idx]

    def _setup(self, tokenizer, fasta, pdb_dir, coord_dir, dssp_dir, graph_dir, rho_dir, is_iedb, max_pad, d_seq, k, radius) :
        ids, seqs_raw = load_fasta(fasta, pdb_dir)
        seqs = [seq.upper() for seq in seqs_raw]

        # amino-acid tokens
        seqs_tok = tokenize(tokenizer, seqs, ids, max_pad).tolist()

        # lengths
        lengths = [len(seq)+1 for seq in seqs]

        # masks
        masks = [pad([1]*len(seq), max_pad) for seq in seqs]

        # labels
        ep_resi = [[int(c.isupper()) for c in seq] for seq in seqs_raw]
        ep_resi_pad = [pad(epr, max_pad) for epr in ep_resi]

        # DSSP features
        dssp_feats_unpad = [get_dssp_feats(dssp_dir, pdb_dir, pdb_id, seq) for pdb_id, seq in zip(ids, seqs)]
        dssp_feats = [pad(entry, max_pad, empty=[0]*11) for entry in dssp_feats_unpad]

        # IEDB flag-embeddings
        iedb_emb_single = [0, 0]
        iedb_emb_single[is_iedb] = 1
        iedb_emb = [pad([iedb_emb_single for i in range(len(seq))], max_pad, empty=[0,0]) for seq in seqs]

        # coordinate embeddings
        coors_unpad = [get_coords(coord_dir, pdb_dir, pdb_id, seq) for pdb_id, seq in zip(ids, seqs)]
        coors = [pad(coor, max_pad, empty=[0,0,0]) for coor in coors_unpad]

        # build graph to be used by egnn, if not already built
        graphs = [build_graph(graph_dir, pdb_id, seq, coord, d_seq, k, radius) for pdb_id, seq, coord in zip(ids, seqs, coors_unpad)]
        adj = [graph[0] for graph in graphs]
        feat = [graph[1] for graph in graphs]

        # surface features (residue depth)
        # TODO optimize
        rho_unpad = [residue_depth(rho_dir, coord, pdb_id, [1., 2., 5., 10., 30.]) for coord, pdb_id in zip(coors_unpad, ids)]
        rho = [pad(rd, max_pad, empty=[0,0,0,0,0]) for rd in rho_unpad]

        return seqs_tok, lengths, masks, coors, rho, adj, feat, dssp_feats, iedb_emb, ep_resi_pad, max_pad, d_seq


