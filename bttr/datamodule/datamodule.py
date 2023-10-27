
import os
import io
import random

from dataclasses import dataclass
from typing import List, Optional, Tuple

import pytorch_lightning as pl
import torch
from PIL import Image
from torch import FloatTensor, LongTensor
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
import lmdb

from .img_aug.transform import get_transform

from .vocab import CROHMEVocab
vocab = CROHMEVocab(charlen=94)

Data = List[Tuple[str, Image.Image, List[str]]]

MAX_SIZE = 15e8  # change here accroading to your GPU memory

@dataclass
class Batch:
    img_bases: List[str]  # [b,]
    imgs: FloatTensor  # [b, 1, H, W]
    mask: LongTensor  # [b, H, W]
    indices: List[List[int]]  # [b, l]
    gts: FloatTensor # [b, 1, H, W]
    
    def __len__(self) -> int:
        return len(self.img_bases)

    def to(self, device) -> "Batch":
        return Batch(
            img_bases=self.img_bases,
            imgs=self.imgs.to(device),
            mask=self.mask.to(device),
            indices=self.indices,
            gts = self.gts.to(device),
        )

def collate_fn(batch):
    return batch[0]

def getbatch(batch):
    fnames = batch[0]
    images_x = batch[1]
    seqs_y = [vocab.label2indices(x) for x in batch[2]]
    gt_x = batch[3]
    heights_x = []
    widths_x = []
    imgs = []
    for img, gt in zip(images_x,gt_x):
        # gt = transforms.ToTensor()(gt) 
        imgs.append(img)
        heights_x.append(img.size(1))
        widths_x.append(img.size(2))
    n_samples = len(heights_x)
    max_height_x = max(heights_x)
    max_width_x = max(widths_x)

    x = torch.zeros(n_samples, 1, max_height_x, max_width_x)
    x_mask = torch.ones(n_samples, max_height_x, max_width_x, dtype=torch.bool)
    for idx, s_x in enumerate(imgs):
        x[idx, :, : s_x.size(1), : s_x.size(2)] = s_x
        x_mask[idx, : s_x.size(1), : s_x.size(2)] = 0
    
    # 载入图片gt：
    imgs_y = torch.zeros(n_samples, 1, 1, 1)
    # for idx, gt in enumerate(gt_x):
    #     gt = transforms.ToTensor()(gt) 
    #     h = gt.size(1)
    #     w = gt.size(2)
    #     imgs_y[idx,:, :h,:w] = gt
    return Batch(fnames, x, x_mask, seqs_y, imgs_y)

            
class CstDataSet(Dataset):
    def __init__(self, datadir: str, datatype: str, batch_size: int, istrain=True,is_aug=True):
        lmdbpath = os.path.join(datadir, datatype + ".lmdb")
        self.env = lmdb.open(lmdbpath, readonly=True)
        self.bs = batch_size
        self.isTrain = istrain
        self.isAug = is_aug
        self.Test500 = True if datatype == "val" else False
        with self.env.begin() as txn:
            cursor = txn.cursor()
            count = sum(1 for _ in cursor)
        self.len = count
        self.transform = get_transform(img_size=[128, 32],augment=self.isAug)

    def __getitem__(self,index):
        
        env = self.env
        if self.isTrain:
            indexes = random.sample(range(int(self.len / self.bs / 2)), self.bs)
        else:
            start_idx = index * self.bs
            end_idx = (index + 1) * self.bs 
            indexes = range(start_idx, end_idx)
        filenames = []
        images = []
        annotations = []
        with env.begin() as txn:
            for item_idx in indexes:
                image_key = f"{item_idx}_image".encode()
                label_key = f"{item_idx}_label".encode()
                imgbuf = txn.get(image_key)
                buf = io.BytesIO(imgbuf)
                img = Image.open(buf).convert('RGB')
                tensor = self.transform(img)
                img = 0.299 * tensor[0] + 0.587 * tensor[1] + 0.114 * tensor[2]
                img = img.view(1,128,32)
                label = txn.get(label_key).decode()
                filenames.append(image_key)
                images.append(img)
                annotations.append(label)
            batch = [filenames, images, annotations, [0] * len(filenames)]
            data = getbatch(batch)
        return data
                
    def __len__(self):
        if self.Test500:
            return 500
        return 5000 if self.isTrain else int(self.len / self.bs / 2)

class CROHMEDatamodule(pl.LightningDataModule):
    def __init__(
        self,
        datapath: str,
        batch_size: int = 8,
        num_workers: int = 0,
        is_aug = False
    ) -> None:
        super().__init__()
        self.datapath = datapath
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.is_aug = is_aug

    def setup(self, stage: Optional[str] = None) -> None:
        datadir = self.datapath
        if stage == "fit" or stage is None:
            self.train_dataset = CstDataSet(datadir, "train", self.batch_size,istrain=True,is_aug=self.is_aug)
            self.val_dataset = CstDataSet(datadir, "val", 1,istrain=True,is_aug=False)
        if stage == "test" or stage is None:
            self.test_dataset = CstDataSet(datadir, "test", 1,istrain=False,is_aug=False)

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            shuffle=True,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            shuffle=False,
            num_workers=self.num_workers,
            collate_fn=collate_fn,
        )


if __name__ == "__main__":
    from argparse import ArgumentParser

    batch_size = 2

    parser = ArgumentParser()
    parser = CROHMEDatamodule.add_argparse_args(parser)

    args = parser.parse_args(["--batch_size", f"{batch_size}"])

    dm = CROHMEDatamodule(**vars(args))
    dm.setup()

    train_loader = dm.train_dataloader()
    for img, mask, tgt, output in train_loader:
        break
