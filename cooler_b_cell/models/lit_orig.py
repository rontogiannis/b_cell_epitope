import pytorch_lightning as pl
import torch

from cooler_b_cell.models.epitope import Epitope
from typing import Callable
from torchmetrics import AUROC, MatthewsCorrCoef, AveragePrecision, F1Score, PrecisionRecallCurve, Precision, Recall
from torch import nn

class EpitopeLit(pl.LightningModule) :
    def __init__(
        self,
        lr: float = 0.001,
        **kwargs,
    ) :
        super().__init__()

        self.lr = lr
        self.model = Epitope(**kwargs)
        self.criterion = nn.BCEWithLogitsLoss() if self.model.output_dim == 1 else nn.CrossEntropyLoss(weight = torch.tensor([.1, .4, .5], dtype=torch.float))

        assert self.model.output_dim in [1, 3]

        self.auc = AUROC(task="multilabel", num_labels=self.model.output_dim) if self.model.output_dim > 1 else AUROC(task="binary")
        self.aupr = AveragePrecision(task="multilabel", num_labels=self.model.output_dim) if self.model.output_dim > 1 else AveragePrecision(task="binary")

        if self.model.output_dim ==1 :
            self.prcurve = PrecisionRecallCurve(task="binary")

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
        out = out.flatten(end_dim=-2 if self.model.output_dim > 1 else -1)
        out = out[mask]
        y = y.flatten(end_dim=-2 if self.model.output_dim > 1 else -1)
        y = y[mask]

        if not training :
            self.auc.update(out, y)
            self.aupr.update(out, y)

            if self.model.output_dim == 1 :
                self.prcurve.update(out, y)

        return self.criterion(out, y.float())

    def on_validation_start(self) :
        self.auc.reset()
        self.aupr.reset()

        if self.model.output_dim == 1 :
            self.prcurve.reset()

    def on_test_start(self) :
        self.auc.reset()
        self.aupr.reset()

        if self.model.output_dim == 1 :
            self.prcurve.reset()

    def training_step(self, batch, batch_idx) :
        loss = self._shared_step(batch, batch_idx, training=True)
        self.log("training/loss", loss.item(), sync_dist=True)
        return loss

    def validation_step(self, batch, batch_idx) :
        self.log("validation/loss", self._shared_step(batch, batch_idx).item(), sync_dist=True)

    def test_step(self, batch, batch_idx) :
        self.log("test/loss", self._shared_step(batch, batch_idx).item(), sync_dist=True)

    def predict_step(self, batch, batch_idx) :
        pid = batch["pdb"]
        tokens = batch["tokens"]
        coord = batch["coord"]
        node_feat = batch["node_feat"]
        edge_feat = batch["edge_feat"]
        mask = batch["mask"]
        graph = batch["graph"]

        out = self.model(tokens, coord, node_feat, edge_feat, mask, graph).squeeze(0)
        out = nn.Softmax(dim=1)(out) if self.model.output_dim > 1 else nn.Sigmoid()(out).squeeze()
        mask = mask.squeeze(0)

        return {
            "pdb": pid[0],
            "pred": out[mask].tolist(),
        }

    def _log_on_end(self, stage) :
        self.log(f"{stage}/auc", self.auc.compute(), sync_dist = True)
        self.log(f"{stage}/aupr", self.aupr.compute(), sync_dist = True)

        if self.model.output_dim == 1 :
            p, r, thresholds = self.prcurve.compute()
            f1 = (2*p*r/(p+r)).nan_to_num(0)[:-1]
            best_threshold = thresholds[torch.argmax(f1)]

            self.log(f"{stage}/f1", f1[torch.argmax(f1)].item(), sync_dist = True)
            self.log(f"{stage}/precision", p[torch.argmax(f1)].item(), sync_dist = True)
            self.log(f"{stage}/recall", r[torch.argmax(f1)].item(), sync_dist = True)
            self.log(f"{stage}/threshold", best_threshold.item(), sync_dist = True)

    def on_validation_epoch_end(self) :
        self._log_on_end("validation")

    def on_test_epoch_end(self) :
        self._log_on_end("test")

