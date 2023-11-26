
import os
import json
from dataclasses import dataclass
from typing import List, Optional, Tuple
from zipfile import ZipFile
from io import BytesIO

import pytorch_lightning as pl
import torch
from PIL import Image
from torch import FloatTensor, LongTensor
from torch.utils.data import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
import requests
import random

from .vocab import CROHMEVocab

import pickle
import numpy as np
import time
import cv2
import lmdb
vocab = CROHMEVocab()

Data = List[Tuple[str, Image.Image, List[str]]]

MAX_SIZE = 15e8  # change here accroading to your GPU memory

def img_aug(img):
    w,h = img.size
    if random.random() > 0.5:
        img = transforms.ToTensor()(img)
        return img
    if random.random() > 0.5:
        img = transforms.Pad(padding=10, fill = 0)(img)
    if random.random() > 0.5:
        ratio = random.random() * 0.2 + 0.8
        img = transforms.Resize(size = [int(img.height * ratio), int(img.width * ratio)])(img)
    ## 加一个丢失信息的resize
    if random.random() > 0.5:
        img = transforms.ColorJitter(brightness=.5, hue=.3)(img)
    if random.random() > 0.5:
        img = transforms.RandomPerspective(distortion_scale=0.1,p=1)(img)
    if random.random() > 0.5:
        img = transforms.RandomRotation(degrees = 10)(img)
    if random.random() > 0.5:
        img = transforms.RandomEqualize()(img)
    img = img.resize((w,h))
    img = transforms.ToTensor()(img)
    return img

def resize_image_with_limit(img, max_width = 256, max_height = 48):
    
    # 获取原始图片的宽和高
    original_width, original_height = img.size

    # 计算缩放比例
    width_ratio = max_width / original_width
    height_ratio = max_height / original_height
    scale_factor = min(width_ratio, height_ratio)
    
    # 计算缩放后的宽和高
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)
    
    # 使用Image.thumbnail函数进行等比例缩放
    img.thumbnail((new_width, new_height), Image.ANTIALIAS)
    
    # 返回缩放后的图片和缩放比例
    return img, scale_factor

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

def getbatch(batch,istrain=False,is_aug=False):
    fnames = batch[0]
    images_x = batch[1]
    seqs_y = [vocab.words2indices(x.decode()) for x in batch[2]]
    gt_x = batch[3]
    heights_x = []
    widths_x = []
    imgs = []
    for img,gt in zip(images_x,gt_x):
        img = img_aug(img) if is_aug else transforms.ToTensor()(img) 
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
        # lmdbpath = "./data/test.lmdb"
        self.env = lmdb.open(lmdbpath, readonly=True)
        self.bs = batch_size
        self.isTrain = istrain
        self.isAug = is_aug
        self.Test500 = True if datatype == "val" else False
        
        with self.env.begin() as txn:
            cursor = txn.cursor()
            count = sum(1 for _ in cursor)
        self.len = count
    
    def getitem_test(self, index):
        env = self.env

        filenames = []
        images = []
        annotations = []

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
                image_value = txn.get(image_key)
                img_stream = BytesIO(image_value)
                img = Image.open(img_stream)

                img = img.resize((256,48), Image.ANTIALIAS)

                label = txn.get(label_key)
                filenames.append(image_key)
                images.append(img)
                annotations.append(label)
            batch = [filenames, images, annotations, [0] * len(filenames)]
            data = getbatch(batch,self.isTrain, self.isAug)
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
