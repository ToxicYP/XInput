from pytorch_lightning import Trainer

from bttr.datamodule import CROHMEDatamodule
from bttr.lit_bttr import LitBTTR

if __name__ == "__main__":
    ckp_path = "./lightning_logs/version_0/checkpoints/epoch=9-step=24999-val_ExpRate=0.3712.ckpt"
    lmdb_test_dir = "./data/sub"
    ckp_paths = []
    lmdb_test_src_path = []
    trainer = Trainer(logger=False)
    dm = CROHMEDatamodule(datapath=lmdb_test_dir)
    model = LitBTTR.load_from_checkpoint(ckp_path)
    print(f"{ckp_path}, TEST DATASET {lmdb_test_dir}")
    trainer.test(model, datamodule=dm)
