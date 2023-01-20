from pytorch_lightning.cli import LightningCLI
from pytorch_lightning.loggers import WandbLogger

def cli_main() :
    cli = LightningCLI(save_config_callback=None)

if __name__ == "__main__" :
    cli_main()
