import pytorch_lightning as pl

from torch.utils.data import Dataset, DataLoader, random_split

from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.utilities.model_summary import ModelSummary

from b_cell.datasets.epitope import EpitopeDataset
from b_cell.models.lit import EpitopeLitModule

# TODO: horrible code
def train(
    model,
    num_epochs,
    train_fasta,
    dev_fasta,
    train_pdb_dir,
    train_coord_dir,
    train_dssp_dir,
    train_graph_dir,
    train_rho_dir,
    dev_pdb_dir,
    dev_coord_dir,
    dev_dssp_dir,
    dev_graph_dir,
    dev_rho_dir,
    checkpoint_dir,
    num_workers=8,
    batch_size=3,
    max_pad=950,
    d_seq=2,
    k=10,
    radius=10.,
    include_iedb=False, # currently does nothing
) :
    auc_callback = ModelCheckpoint(
        monitor="validation/auc",
        dirpath=checkpoint_dir,
        filename="best-auc",
        auto_insert_metric_name=False,
        mode="max",
    )

    top_callback = ModelCheckpoint(
        monitor="validation/top_acc",
        dirpath=checkpoint_dir,
        filename="best-top",
        auto_insert_metric_name=False,
        mode="max",
    )

    top_k_callback = ModelCheckpoint(
        monitor="validation/top_acc_k",
        dirpath=checkpoint_dir,
        filename="best-top-k",
        auto_insert_metric_name=False,
        mode="max",
    )

    ModelSummary(model)

    train_dataset = EpitopeDataset(
        tokenizer=model.model.esm_model.alphabet,
        fasta=train_fasta,
        pdb_dir=train_pdb_dir,
        coord_dir=train_coord_dir,
        dssp_dir=train_dssp_dir,
        graph_dir=train_graph_dir,
        rho_dir=train_rho_dir,
        is_iedb=0,
        max_pad=max_pad,
        d_seq=d_seq,
        k=k,
        radius=radius,
    )

    dev_dataset = EpitopeDataset(
        tokenizer=model.model.esm_model.alphabet,
        fasta=dev_fasta,
        pdb_dir=dev_pdb_dir,
        coord_dir=dev_coord_dir,
        dssp_dir=dev_dssp_dir,
        graph_dir=dev_graph_dir,
        rho_dir=dev_rho_dir,
        is_iedb=0,
        max_pad=max_pad,
        d_seq=d_seq,
        k=k,
        radius=radius,
    )

    print(f"{len(train_dataset)=}, {len(dev_dataset)=}")


    #train_size = len(train_dataset)

    #train_split, dev_split = random_split(
    #    dataset=train_dataset,
    #    lengths=[train_size-train_size//10, train_size//10],
    #)

    train_split, dev_split = train_dataset, dev_dataset

    train_loader = DataLoader(train_split, num_workers=num_workers, batch_size=batch_size, shuffle=True)
    dev_loader = DataLoader(dev_split, num_workers=num_workers, batch_size=batch_size, shuffle=False)

    wandb_logger = WandbLogger(project="b_cell top metric experiments")

    trainer = pl.Trainer(
        logger=wandb_logger,
        callbacks=[auc_callback, top_callback, top_k_callback],
        accelerator="gpu",
        devices=-1,
        strategy="ddp",
        log_every_n_steps=10,
        max_epochs=num_epochs,
    )

    trainer.fit(model, train_loader, dev_loader)

    return [
        auc_callback.best_model_path,
        top_callback.best_model_path,
        top_k_callback.best_model_path,
    ]

def test(
    model_checkpoint,
    test_fasta,
    test_pdb_dir,
    test_coord_dir,
    test_dssp_dir,
    test_graph_dir,
    test_rho_dir,
    num_workers=8,
    batch_size=3,
    max_pad=950,
    d_seq=2,
    k=10,
    radius=10.,
) :
    model = EpitopeLitModule.load_from_checkpoint(model_checkpoint, map_location="cpu")

    test_dataset = EpitopeDataset(
        tokenizer=model.model.esm_model.alphabet,
        fasta=test_fasta,
        pdb_dir=test_pdb_dir,
        coord_dir=test_coord_dir,
        dssp_dir=test_dssp_dir,
        graph_dir=test_graph_dir,
        rho_dir=test_rho_dir,
        is_iedb=0,
        max_pad=max_pad,
        d_seq=d_seq,
        k=k,
        radius=radius,
    )

    test_loader = DataLoader(test_dataset, num_workers=num_workers, batch_size=batch_size, shuffle=False)

    tester = pl.Trainer(
        accelerator="gpu",
        devices=1,
        enable_checkpointing=False,
        log_every_n_steps=10,
    )

    tester.test(model, test_loader)

def predict(
    model_checkpoint,
    test_fasta,
    test_pdb_dir,
    test_coord_dir,
    test_dssp_dir,
    test_graph_dir,
    test_rho_dir,
    num_workers=8,
    max_pad=950,
    d_seq=2,
    k=10,
    radius=10.,
) :
    model = EpitopeLitModule.load_from_checkpoint(model_checkpoint, map_location="cpu")

    test_dataset = EpitopeDataset(
        tokenizer=model.model.esm_model.alphabet,
        fasta=test_fasta,
        pdb_dir=test_pdb_dir,
        coord_dir=test_coord_dir,
        dssp_dir=test_dssp_dir,
        graph_dir=test_graph_dir,
        rho_dir=test_rho_dir,
        is_iedb=0,
        max_pad=max_pad,
        d_seq=d_seq,
        k=k,
        radius=radius,
    )

    test_loader = DataLoader(test_dataset, num_workers=num_workers, batch_size=1, shuffle=False)

    tester = pl.Trainer(
        accelerator="gpu",
        devices=1,
        enable_checkpointing=False,
        log_every_n_steps=10,
    )

    return tester.predict(model, test_loader)
