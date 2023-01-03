import pytorch_lightning as pl
import torch

from b_cell.models.epitope import EpitopePredictionModel
from typing import Callable
from torchmetrics import AUROC

class EpitopeLitModule(pl.LightningModule) :
    def __init__(
        self,
        criterion: Callable,
        lr: float,
        k: int = 10,
        use_topk_loss: bool = False,
        **kwargs,
    ) :
        super().__init__()

        self.save_hyperparameters()

        self.k = k
        self.use_topk_loss = use_topk_loss
        self.criterion = criterion
        self.model = EpitopePredictionModel(**kwargs)

        self.auc = AUROC()

        self.yes = 0
        self.yes_k = 0
        self.all = 0

    def configure_optimizers(self) :
        params = list(self.model.named_parameters())

        # if we are fine-tuning the language model, make sure to use small lr
        grouped_parameters = [
            {"params": [p for n, p in params if "esm" in n and p.requires_grad], "lr": self.hparams.lr/100.},
            {"params": [p for n, p in params if "esm" not in n], "lr": self.hparams.lr},
        ]

        return torch.optim.Adam(grouped_parameters, lr=self.hparams.lr)

    def gradient_norm(self) :
        total_norm = 0

        for p in self.model.parameters() :
            if p.requires_grad and p.grad != None :
                total_norm += p.grad.detach().data.norm(2).item()**2

        return total_norm**.5

    def _shared_step(self, batch, batch_idx, update_auc=True, update_top_metric=True) :
        params, mask, y = batch
        out = self.model(params, mask).squeeze(-1)
        out_masked = out - 99*(~mask).float()

        # top-k loss
        top_k_obj = torch.topk(out_masked, self.k, dim=-1)
        top_k_idx = top_k_obj.indices
        top_k_val = top_k_obj.values
        y_top_k = torch.gather(y, -1, top_k_idx)

        if self.use_topk_loss :
            loss = self.criterion(top_k_val, y_top_k.float())

        # evaluation metrics
        top_idx = torch.argmax(out_masked, dim=-1)
        top_y = torch.gather(y, -1, top_idx.unsqueeze(-1)).squeeze(-1)
        top_y_k = torch.max(torch.gather(y, -1, top_k_idx), dim=-1).values

        # simple loss calculation
        mask = mask.flatten()
        out = out.flatten()
        out = torch.masked_select(out, mask)
        y = y.flatten()
        y = torch.masked_select(y, mask)

        if not self.use_topk_loss :
            loss = self.criterion(out, y.float())

        # update metrics
        if update_auc :
            self.auc.update(out, y)

        if update_top_metric :
            self.yes += torch.sum(top_y)
            self.yes_k += torch.sum(top_y_k)
            self.all += top_y.shape[0]

        return loss

    def on_validation_start(self) :
        self.auc.reset()
        self.yes = 0
        self.yes_k = 0
        self.all = 0

    def on_test_start(self) :
        self.auc.reset()
        self.yes = 0
        self.yes_k = 0
        self.all = 0

    def training_step(self, batch, batch_idx) :
        loss = self._shared_step(batch, batch_idx, update_auc=False)
        self.log("training/loss", loss.item())
        self.log("training/grad_norm", self.gradient_norm())

        return loss

    def validation_step(self, batch, batch_idx) :
        self.log("validation/loss", self._shared_step(batch, batch_idx).item())

    def test_step(self, batch, batch_idx) :
        self.log("test/loss", self._shared_step(batch, batch_idx).item())

    def on_validation_epoch_end(self) :
        self.log("validation/auc", self.auc.compute())
        self.log("validation/top_acc", self.yes/self.all)
        self.log("validation/top_acc_k", self.yes_k/self.all)

    def on_test_epoch_end(self) :
        self.log("test/auc", self.auc.compute())
        self.log("test/top_acc", self.yes/self.all)
        self.log("test/top_acc_k", self.yes_k/self.all)

