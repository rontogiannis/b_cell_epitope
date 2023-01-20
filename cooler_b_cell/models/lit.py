import pytorch_lightning as pl
import torch

from cooler_b_cell.models.epitope import Epitope
from typing import Callable
from torchmetrics import AUROC
from torch import nn

class EpitopeLit(pl.LightningModule) :
    def __init__(
        self,
        lr: float = 0.001,
        **kwargs,
    ) :
        super().__init__()

        self.criterion = nn.CrossEntropyLoss(weight = torch.tensor([.15, .35, .5], dtype=torch.float))
        self.lr = lr
        self.model = Epitope(**kwargs)
        self.auc = AUROC(task="multiclass", num_classes=3)

    def configure_optimizers(self) :
        return torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=self.lr)

    def _shared_step(self, batch, batch_idx, training=False) :
        tokens = batch["tokens"]
        coord = batch["coord"]
        node_feat = batch["node_feat"]
        edge_feat = batch["edge_feat"]
        mask = batch["mask"]
        graph = batch["graph"]
        y = batch["y"]

        out = self.model(tokens, coord, node_feat, edge_feat, mask, graph)

        mask = mask.flatten()
        out = out.flatten(end_dim=-2)
        out = out[mask]
        y = y.flatten()
        y = y[mask]

        if not training :
            self.auc.update(out, y)

        return self.criterion(out, y)

    def on_validation_start(self) :
        self.auc.reset()

    def on_test_start(self) :
        self.auc.reset()

    def training_step(self, batch, batch_idx) :
        loss = self._shared_step(batch, batch_idx, training=True)
        self.log("training/loss", loss.item(), sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx) :
        self.log("validation/loss", self._shared_step(batch, batch_idx).item(), sync_dist=True)

    def test_step(self, batch, batch_idx) :
        self.log("test/loss", self._shared_step(batch, batch_idx).item(), sync_dist=True)

    def on_validation_epoch_end(self) :
        self.log("validation/auc", self.auc.compute(), sync_dist=True)

    def on_test_epoch_end(self) :
        self.log("test/auc", self.auc.compute(), sync_dist=True)

