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
        k: int = 10,
        **kwargs,
    ) :
        super().__init__()

        self.yes = 0
        self.yes_k = 0
        self.all = 0

        self.lr = lr
        self.k = k
        self.model = Epitope(**kwargs)
        self.criterion = nn.BCEWithLogitsLoss() if self.model.output_dim == 1 else nn.CrossEntropyLoss(weight = torch.tensor([.1, .4, .5], dtype=torch.float))
        # self.criterion2 = nn.BCEWithLogitsLoss()

        assert self.model.output_dim in [1, 3]

        self.auc = AUROC(task="multilabel", num_labels=self.model.output_dim) if self.model.output_dim > 1 else AUROC(task="binary")
        self.aupr = AveragePrecision(task="multilabel", num_labels=self.model.output_dim) if self.model.output_dim > 1 else AveragePrecision(task="binary")

        if self.model.output_dim ==1 :
            self.prcurve = PrecisionRecallCurve(task="binary")

    def configure_optimizers(self) :
        return torch.optim.Adam([p for p in self.model.parameters() if p.requires_grad], lr=self.lr)

    def _additional_loss(self, y, out, mask, i) :
        out_epitope = torch.sum(out[:,:,i:], 2)
        out_epitope = out_epitope - 99*(~mask).float()

        top_k_obj = torch.topk(out_epitope, self.k, dim=-1)
        top_k_idx = top_k_obj.indices

        top_k_idx_unsqueezed = torch.empty_like(top_k_idx).copy_(top_k_idx)
        top_k_idx_unsqueezed = torch.unsqueeze(top_k_idx_unsqueezed, -1)
        top_k_idx_unsqueezed = torch.cat((top_k_idx_unsqueezed, top_k_idx_unsqueezed, top_k_idx_unsqueezed), -1)

        out_all_top_k = torch.gather(out, 1, top_k_idx_unsqueezed)
        out_all_top_k = out_all_top_k.flatten(end_dim=-2 if self.model.output_dim > 1 else -1)

        y_all_top_k = torch.gather(y, 1, top_k_idx_unsqueezed)
        y_all_top_k = y_all_top_k.flatten(end_dim=-2 if self.model.output_dim > 1 else -1)

        additional_loss = self.criterion(out_all_top_k.flatten(end_dim=-2), y_all_top_k.flatten(end_dim=-2).float())

        return additional_loss, out_epitope, top_k_idx

    def _shared_step(self, batch, batch_idx, training=False) :
        tokens = batch["tokens"]
        coord = batch["coord"]
        node_feat = batch["node_feat"]
        edge_feat = batch["edge_feat"]
        mask = batch["mask"]
        graph = batch["graph"]
        y = batch["y"]

        out = self.model(tokens, coord, node_feat, edge_feat, mask, graph)

        add1, out_epitope, top_k_idx = self._additional_loss(y, out, mask, 1)
        add2, _, _ = self._additional_loss(y, out, mask, 2)

        y_epitope = torch.sum(y[:,:,1:], 2).bool()

        mask = mask.flatten()
        out = out.flatten(end_dim=-2 if self.model.output_dim > 1 else -1)
        out = out[mask]
        y = y.flatten(end_dim=-2 if self.model.output_dim > 1 else -1)
        y = y[mask]

        main_loss = self.criterion(out, y.float())

        if not training :
            top_idx = torch.argmax(out_epitope, dim=-1)
            top_y = torch.gather(y_epitope, -1, top_idx.unsqueeze(-1)).squeeze(-1)
            top_y_k = torch.max(torch.gather(y_epitope, -1, top_k_idx), dim=-1).values

            self.yes += torch.sum(top_y)
            self.yes_k += torch.sum(top_y_k)
            self.all += top_y.shape[0]

            self.auc.update(out, y)
            self.aupr.update(out, y)

            if self.model.output_dim == 1 :
                self.prcurve.update(out, y)

        return main_loss

    def on_validation_start(self) :
        self.auc.reset()
        self.aupr.reset()

        self.yes = 0
        self.yes_k = 0
        self.all = 0

        if self.model.output_dim == 1 :
            self.prcurve.reset()

    def on_test_start(self) :
        self.auc.reset()
        self.aupr.reset()

        self.yes = 0
        self.yes_k = 0
        self.all = 0

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

        self.log(f"{stage}/top_1_acc", self.yes/self.all, sync_dist = True)
        self.log(f"{stage}/top_k_acc", self.yes_k/self.all, sync_dist = True)

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

