import json
import torch
from pytorch_lightning.callbacks import BasePredictionWriter

def load_jsonl(path) :
    with open(path, "r") as f :
        json_l = [json.loads(l.strip()) for l in f.readlines()]
        f.close()
    return json_l


class Writer(BasePredictionWriter):
    def __init__(self, path, write_interval):
        super().__init__(write_interval)

        self.path = path

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        prediction,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx
    ):
        pass

    def write_on_epoch_end(
        self,
        trainer,
        pl_module,
        predictions,
        batch_indices
    ):
        preds = predictions[0] # one line per epoch aparently?
        with open(self.path, "w") as f :
            for line in preds :
                json.dump(line, f)
                f.write("\n")
            f.close()
