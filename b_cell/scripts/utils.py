import lzma
import pickle
import math
import os
import traceback
import sys
import Bio

from sklearn.neighbors import NearestNeighbors, KDTree
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import DSSP
from pylcs import lcs_sequence_idx

d3to1 = {
    "ALA": "A",
    "ARG": "R",
    "ASN": "N",
    "CYS": "C",
    "CSS": "C",
    "ASP": "D",
    "SER": "S",
    "GLN": "Q",
    "LYS": "K",
    "ILE": "I",
    "PRO": "P",
    "THR": "T",
    "PHE": "F",
    "GLY": "G",
    "HIS": "H",
    "LEU": "L",
    "TRP": "W",
    "VAL": "V",
    "GLU": "E",
    "TYR": "Y",
    "MET": "M",
    "MSE": "M",
}

amino2int = {
    "A": 0,
    "R": 1,
    "N": 2,
    "D": 3,
    "C": 4,
    "Q": 5,
    "E": 6,
    "G": 7,
    "H": 8,
    "I": 9,
    "L": 10,
    "K": 11,
    "M": 12,
    "F": 13,
    "P": 14,
    "S": 15,
    "T": 16,
    "W": 17,
    "Y": 18,
    "V": 19,
    "X": 20,
}

ss_map = {
    "H": 0,
    "B": 1,
    "E": 2,
    "G": 3,
    "I": 4,
    "T": 5,
    "S": 6,
    "-": 7,
}

# utility functions
def load_pickle(directory, pdb_id) :
    if os.path.isfile(directory.format(pdb_id)) :
        with lzma.open(directory.format(pdb_id), "rb") as f :
            stuff = pickle.load(f)
            f.close()
        return stuff
    return None

def dump_pickle(directory, pdb_id, stuff) :
    with lzma.open(directory.format(pdb_id), "wb") as f :
        pickle.dump(stuff, f)
        f.close()

def pad(seq, max_pad, empty=0) :
    return [empty]+seq+(max_pad-len(seq)+1)*[empty]

def tokenize(tokenizer, seqs, ids, max_pad) :
    seq_with_id = list(zip(ids, seqs)) + [("dummy", "<mask>"*max_pad)]
    _, _, batch_tokens = tokenizer.get_batch_converter()(seq_with_id)
    return batch_tokens[:-1]

def _sq_norm(xi, xj) :
    a = xi[0]-xj[0]
    b = xi[1]-xj[1]
    c = xi[2]-xj[2]
    return a*a + b*b + c*c

def _norm(xi) :
    a = xi[0]
    b = xi[1]
    c = xi[2]
    return math.sqrt(a*a + b*b + c*c)

def _sum_vecs(vecs) :
    chi, psi, zed = 0, 0, 0
    for vec in vecs :
        chi += vec[0]
        psi += vec[1]
        zed += vec[2]
    return chi, psi, zed

def _m_diff(m, xi, xj) :
    a = xi[0]-xj[0]
    b = xi[1]-xj[1]
    c = xi[2]-xj[2]
    return [m*a, m*b, m*c]

# residue depth
def residue_depth(rho_dir, coord, pdb_id, lambdas, kappa=10) :
    stuff = load_pickle(rho_dir, pdb_id)

    if stuff != None :
        return stuff

    n = len(coord)
    ls = len(lambdas)

    _, NN = NearestNeighbors(n_neighbors=kappa+1, algorithm="ball_tree").fit(coord).kneighbors(coord)

    pairwise = [[[math.exp(-_sq_norm(coord[i], coord[NN[i][j]])/lam)
                    for j in range(1,kappa+1)]
                    for i in range(n)]
                    for lam in lambdas]

    sums = [[sum(pairwise[l][i])
                for i in range(n)]
                for l in range(ls)]

    w = [[[pairwise[l][i][j]/sums[l][i]
            for j in range(kappa)]
            for i in range(n)]
            for l in range(ls)]

    numer = [[_norm(_sum_vecs([_m_diff(w[l][i][j-1], coord[i], coord[NN[i][j]])
                for j in range(1,kappa+1)]))
                for i in range(n)]
                for l in range(ls)]

    denom = [[sum([w[l][i][j-1]*_norm(_m_diff(1, coord[i], coord[NN[i][j]]))
                for j in range(1,kappa+1)])
                for i in range(n)]
                for l in range(ls)]

    rho = [[numer[l][i]/denom[l][i]
            for l in range(ls)]
            for i in range(n)]

    dump_pickle(rho_dir, pdb_id, rho)

    return rho

# graph
def build_graph(graph_dir, pdb_id, seq, coord, d_seq=2, k=10, radius=10.) :
    stuff = load_pickle(graph_dir, pdb_id)

    if stuff != None :
        return stuff

    kd = KDTree(coord, leaf_size=4)

    knn = kd.query(coord, k=k+1, return_distance=False)
    bal = kd.query_radius(coord, radius, return_distance=False)

    edge_dim = 21+21+2*d_seq+2
    n = len(seq)

    adj = [[],[]]
    feat = [[],[],[]]
    edge = []

    for i in range(n) :
        # first handle k-nearest neighbors
        for j in knn[i] :
            if j == i :
                continue

            edge.append((i, j))

            feat[0].append(i+1)
            feat[1].append(j+1)
            feat[2].append(21+21+2*d_seq)

        # then look in a circle certered at i
        for j in bal[i] :
            if j == i :
                continue

            edge.append((i, j))

            feat[0].append(i+1)
            feat[1].append(j+1)
            feat[2].append(21+21+2*d_seq+1)

        # finally account for sequential connections
        for j in range(max(0, i-d_seq), min(n, i+d_seq+1)) :
            if j == i :
                continue

            edge.append((i, j))

            feat[0].append(i+1)
            feat[1].append(j+1)
            feat[2].append(21+21+d_seq+i-j-(1 if i>j else 0))

    # the, add all edges to adjacency list and add amino-acid features
    for i, j in set(edge) :
        adj[0].append(i+1)
        adj[1].append(j+1)

        feat[0].append(i+1)
        feat[1].append(j+1)
        feat[2].append(amino2int[seq[i]])

        feat[0].append(i+1)
        feat[1].append(j+1)
        feat[2].append(21+amino2int[seq[j]])

    dump_pickle(graph_dir, pdb_id, [adj, feat])

    return [adj, feat]

# DSSP features
def get_dssp_feats(dssp_dir, pdb_dir, pdb_id, known_seq) :
    stuff = load_pickle(dssp_dir, pdb_id)

    if stuff != None :
        return stuff

    p = PDBParser()
    structure = p.get_structure(pdb_id, pdb_dir.format(pdb_id))
    model = structure[0]

    try :
        dssp = DSSP(model, pdb_dir.format(pdb_id), dssp="mkdssp")
    except Bio.PDB.PDBExceptions.PDBException:
        traceback.print_exc()
        print("Culprit:", pdb_id)
        sys.exit(0)

    out = []
    out_aligned = []
    seq_dssp = ""

    # e.g. result of dssp:
    # (829, 'E', '-', 0.29381443298969073, -114.1, 360.0, -2, -0.9, -219, -0.3, -11, -0.2, -221, -0.1)

    for residue in dssp :
        seq_dssp = seq_dssp + residue[1]
        ss_emb = [0]*8
        ss_emb[ss_map[residue[2]]] = 1 # secondary structure
        out.append(ss_emb+[residue[3]*100, residue[4], residue[5]]) # relative ASA, phi, psi

    # align the sequence obtained by dssp with the known sequence, padding as needed
    idx = lcs_sequence_idx(known_seq, seq_dssp)

    for i in idx :
        if i == -1 :
            out_aligned.append([0]*11)
        else :
            out_aligned.append(out[i])

    assert len(out_aligned) == len(known_seq), f"{pdb_id}, {len(out_aligned)=} != {len(known_seq)=}"

    dump_pickle(dssp_dir, pdb_id, out_aligned)

    return out_aligned

# coordinate extraction from PDB file
def centroid(residue) :
    l = [atom.coord for atom in residue]
    return sum(l)/len(l)

def amino1(a) :
    if a[:-1] == "CS" :
        return "C"
    return d3to1[a] if a in d3to1 else "X"

def get_coords(coord_dir, pdb_dir, pdb_id, known_seq) :
    stuff = load_pickle(coord_dir, pdb_id)

    if stuff != None :
        return stuff

    p = PDBParser()
    structure = p.get_structure(pdb_id, pdb_dir.format(pdb_id))
    model = structure[0]
    chain_id = pdb_id.split("_")[1]
    out = []
    out_aligned = []
    seq_pdb = ""

    for residue in model[chain_id] :
        seq_pdb = seq_pdb + amino1(residue.resname)
        res_coord = residue["CA"].coord.tolist() if "CA" in residue else centroid(residue).tolist()
        out.append(res_coord)

    idx = lcs_sequence_idx(known_seq, seq_pdb)

    for i in idx :
        if i == -1 :
            out_aligned.append([1e-3]*3)
            print(f"Warning: missing residue coordinates for {pdb_id}, replacing with [0, 0, 0]\n{known_seq}\n{seq_pdb}")
        else :
            out_aligned.append(out[i])

    assert len(out_aligned) == len(known_seq), f"{pdb_id}, {len(out_aligned)=} != {len(known_seq)=}"

    dump_pickle(coord_dir, pdb_id, out_aligned)

    return out_aligned

def pick_correct_file(pdb_dir, pdb_id) :
    struct_id, chain_id = pdb_id.split("_")
    is_lower = os.path.isfile(pdb_dir.format(struct_id+"_"+chain_id.lower()))
    is_upper = os.path.isfile(pdb_dir.format(struct_id+"_"+chain_id.upper()))
    if chain_id.isalpha() and ((is_lower and is_upper) or not (is_lower or is_upper)) :
        assert 0, f"{is_lower=}, {is_upper=}, {struct_id+chain_id.lower()}, {struct_id+chain_id.upper()}"
    chain_id = chain_id.lower() if is_lower else chain_id.upper()
    return struct_id + "_" + chain_id

def load_fasta(fasta, pdb_dir="") :
    with open(fasta, "r") as f :
        lines = [l.strip() for l in f.readlines()]
        f.close()

    seqs = [l for l in lines[1::2]]
    ids = [(pick_correct_file(pdb_dir, l[1:]) if pdb_dir != "" else l[1:]) for l in lines[::2]]

    return ids, seqs
