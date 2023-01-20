import pytorch_lightning as pl
from cooler_b_cell.datasets.epitope import Epitope
from torch.utils.data import DataLoader

class EpitopeLit(pl.LightningDataModule) :
    def __init__(
        self,
        train_path: str = "",
        dev_path: str = "",
        test_path: str = "",
        batch_size: int = 8,
        esm_model_name: str = "esm2_t30_150M_UR50D",
        padded_length: int = 1024,
    ) :
        super().__init__()

        self.train_path = train_path
        self.dev_path = dev_path
        self.test_path = test_path
        self.batch_size = batch_size
        self.esm_model_name = esm_model_name
        self.padded_length = padded_length

    def setup(self, stage: str) :
        if stage == "fit" :
            self.train_set = Epitope(self.train_path, self.esm_model_name, self.padded_length)
            self.dev_set = Epitope(self.dev_path, self.esm_model_name, self.padded_length)
        elif stage == "test" or stage == "predict" :
            self.test_set = Epitope(self.test_path, self.esm_model_name, self.padded_length)

    def train_dataloader(self):
        return DataLoader(self.train_set, batch_size=self.batch_size, num_workers=10, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.dev_set, batch_size=self.batch_size, num_workers=10, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_set, batch_size=1, num_workers=10, shuffle=False)

    def predict_dataloader(self):
        return DataLoader(self.test_set, batch_size=1, num_workers=10, shuffle=False)

