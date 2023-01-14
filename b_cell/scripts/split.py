from b_cell.scripts.utils import load_fasta
from numpy.random import choice
from numpy import array
import os

dire = "/data/scratch/aronto/b_cell_epitope/b_cell/data/GraphBepi/train.fasta"
dirt = "/data/scratch/aronto/b_cell_epitope/b_cell/data/GraphBepi/train_split.fasta"
dird = "/data/scratch/aronto/b_cell_epitope/b_cell/data/GraphBepi/dev_split.fasta"

pids, seqs = load_fasta(dire)
n = len(pids)

dev_idx = choice(array(range(n)), size=n//10, replace=False)

with open(dird, "w") as f :
    ll = []
    for i in dev_idx :
        ll.append(pids[i])
        f.write(">" + pids[i] + "\n")
        f.write(seqs[i] + "\n")
    for i in sorted(ll) :
        print(i, os.path.isfile("/data/scratch/aronto/b_cell_epitope/b_cell/data/GraphBepi/pdb/"+i.lower()+".pdb"))
    f.close()

with open(dirt, "w") as f :
    for i in range(n) :
        if i not in dev_idx :
            f.write(">" + pids[i] + "\n")
            f.write(seqs[i] + "\n")
    f.close()



