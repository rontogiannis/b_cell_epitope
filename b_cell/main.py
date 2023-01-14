import pytorch_lightning as pl
import argparse
import json

from torch import nn

from b_cell.models.lit import EpitopeLitModule
from b_cell.scripts.train_test import train, test, predict

HOME = "/data/scratch/aronto/b_cell_epitope/"

FASTA = HOME+"b_cell/data/{}/{}.fasta"
PDB_DIR = HOME+"b_cell/data/{}/pdb/{}.pdb"
COORD_DIR = HOME+"b_cell/data/{}/coord/{}.xz"
DSSP_DIR = HOME+"b_cell/data/{}/dssp/{}.xz"
GRAPH_DIR = HOME+"b_cell/data/{}/graph/{}.xz"
RHO_DIR = HOME+"b_cell/data/{}/rho/{}.xz"
CHECKPOINTS = HOME+"b_cell/checkpoints/"

# constants regarding the graph generation, don't worry about them
D_SEQ = 2
K = 10
RADIUS = 10.

MAX_PAD = 950
NUM_EPOCHS = 500
NUM_WORKERS = 8
BATCH_SIZE = 3 # x the number of GPUs

LR = 1e-5
SMALL_LR = 1e-6

esm_models = {
    "3B": {"name": "esm2_t36_3B_UR50D", "layer_cnt": 36, "dim": 2560},
    "150M": {"name": "esm2_t30_150M_UR50D", "layer_cnt": 30, "dim": 640},
    "35M": {"name": "esm2_t12_35M_UR50D", "layer_cnt": 12, "dim": 480},
    "8M": {"name": "esm2_t6_8M_UR50D", "layer_cnt": 6, "dim": 320},
}

ESM_MODEL_NAME = "150M"

def setup_cmd() :
    parser = argparse.ArgumentParser()

    parser.add_argument("--test", help="test checkpointed model on provided dataset", type=str, default="")
    parser.add_argument("--predict", help="make predictions", type=str, default="")
    parser.add_argument("--train", help="train a model (prioritized over testing, cannot do both at the same time)", type=str, default="")
    parser.add_argument("--dev", help="choose a dev dataset", type=str, default="")
    parser.add_argument("--egnn", help="include equivariant GNN as part of the architecture", action="store_true")
    parser.add_argument("--rho", help="include residue depth as part of the embeddings", action="store_true")
    parser.add_argument("--iedb", help="include IEDB data in training", action="store_true")
    parser.add_argument("--dssp", help="use DSSP embeddings (relative ASA, secondary structure, angles)", action="store_true")
    parser.add_argument("--seed", help="set the seed (default 13)", type=int, default=137)
    parser.add_argument("--pretrained", help="choose a pre-trained model to load", type=str, default="")
    parser.add_argument("--checkpoint", help="choose a checkpoint for testing", type=str, default=CHECKPOINTS+"best.ckpt")
    parser.add_argument("--rnn", help="include a multi-layer RNN as part of the architecture", action="store_true")

    args = vars(parser.parse_args())

    return args

if __name__ == "__main__" :
    args = setup_cmd()

    pl.seed_everything(args["seed"], workers=True)

    if args["train"] != "" :
        if args["pretrained"] == "" :
            model = EpitopeLitModule(
                criterion=nn.BCELoss(),
                lr=LR,
                use_topk_loss=False,
                esm_model_name=esm_models[ESM_MODEL_NAME]["name"],
                esm_layer_cnt=esm_models[ESM_MODEL_NAME]["layer_cnt"],
                esm_dim=esm_models[ESM_MODEL_NAME]["dim"],
                use_egnn=args["egnn"],
                use_rho=args["rho"],
                use_iedb=False, # args["iedb"],
                use_dssp=args["dssp"],
                use_rnn=args["rnn"],
                egnn_dim=256,
                egnn_edge_dim=21+21+2*D_SEQ+2,
                egnn_nn=10,
                egnn_layers=2,
                mlp_hidden_dim=128,
                dropout=0.25,
                finetune_lm=False,
                total_padded_length=MAX_PAD+2,
                rnn_hidden_dim=128,
                rnn_num_layers=3,
                rnn_bidirectional=True,
                use_minifold=True,
                finetune_minifold=True,
            )
        else :
            print("Loading pretrained model from {}".format(args["pretrained"]))
            model = EpitopeLitModule.load_from_checkpoint(args["pretrained"], map_location="cpu")
            model.hparams.lr = SMALL_LR
            #model.use_topk_loss = True
            #model.model.finetune_mlp_only = True

            #for p in model.model.parameters() :
            #    p.requires_grad = False

            #for p in model.model.mlp.parameters() :
            #    p.requires_grad = True

        # TODO horrible code
        checkpoint_paths = train(
            model=model,
            num_epochs=NUM_EPOCHS,
            train_fasta=FASTA.format(args["train"], "train_split"),
            dev_fasta=FASTA.format(args["train"] if args["dev"] == "" else args["dev"], "dev_split"),
            train_pdb_dir=PDB_DIR.format(args["train"], "{}"),
            train_coord_dir=COORD_DIR.format(args["train"], "{}"),
            train_dssp_dir=DSSP_DIR.format(args["train"], "{}"),
            train_graph_dir=GRAPH_DIR.format(args["train"], "{}"),
            train_rho_dir=RHO_DIR.format(args["train"], "{}"),
            dev_pdb_dir=PDB_DIR.format(args["train"] if args["dev"] == "" else args["dev"], "{}"),
            dev_coord_dir=COORD_DIR.format(args["train"] if args["dev"] == "" else args["dev"], "{}"),
            dev_dssp_dir=DSSP_DIR.format(args["train"] if args["dev"] == "" else args["dev"], "{}"),
            dev_graph_dir=GRAPH_DIR.format(args["train"] if args["dev"] == "" else args["dev"], "{}"),
            dev_rho_dir=RHO_DIR.format(args["train"] if args["dev"] == "" else args["dev"], "{}"),
            checkpoint_dir=CHECKPOINTS,
            num_workers=NUM_WORKERS,
            batch_size=BATCH_SIZE,
            max_pad=MAX_PAD,
            d_seq=D_SEQ,
            k=K,
            radius=RADIUS,
            include_iedb=False, # currently does nothing
        )

        print("Best checkpoints saved at\n{}".format("\n".join(checkpoint_paths)))
    elif args["test"] != "" :
        test(
            model_checkpoint=args["checkpoint"],
            test_fasta=FASTA.format(args["test"], "test"),
            test_pdb_dir=PDB_DIR.format(args["test"], "{}"),
            test_coord_dir=COORD_DIR.format(args["test"], "{}"),
            test_dssp_dir=DSSP_DIR.format(args["test"], "{}"),
            test_graph_dir=GRAPH_DIR.format(args["test"], "{}"),
            test_rho_dir=RHO_DIR.format(args["test"], "{}"),
            num_workers=NUM_WORKERS,
            batch_size=BATCH_SIZE,
            max_pad=MAX_PAD,
            d_seq=D_SEQ,
            k=K,
            radius=RADIUS,
        )
    elif args["predict"] != "" :
        predictions = predict(
            model_checkpoint=args["checkpoint"],
            test_fasta=FASTA.format(args["predict"], "test_split"),
            test_pdb_dir=PDB_DIR.format(args["predict"], "{}"),
            test_coord_dir=COORD_DIR.format(args["predict"], "{}"),
            test_dssp_dir=DSSP_DIR.format(args["predict"], "{}"),
            test_graph_dir=GRAPH_DIR.format(args["predict"], "{}"),
            test_rho_dir=RHO_DIR.format(args["predict"], "{}"),
            num_workers=NUM_WORKERS,
            max_pad=MAX_PAD,
            d_seq=D_SEQ,
            k=K,
            radius=RADIUS,
        )

        with open("predictions.json", "w") as f :
            json.dump(predictions, f)
            f.close()
