from pytorch_lightning.utilities.cli import LightningCLI

from bttr.datamodule import CROHMEDatamodule
from bttr.lit_bttr import LitBTTR

import os

# os.environ['CUDA_VISIBLE_DEVICES'] = "0,1"
if __name__=="__main__":
    cli = LightningCLI(LitBTTR, CROHMEDatamodule)
