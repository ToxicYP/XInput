from typing import List, Tuple

import editdistance
import torch
import torch.nn.functional as F
from einops import rearrange
from torch import LongTensor
from torchmetrics import Metric

from bttr.datamodule import vocab

import matplotlib.font_manager as mfm
from matplotlib import mathtext
from PIL import Image
import numpy as np
from io import BytesIO
import random
import json

class Hypothesis:
    seq: List[int]
    score: float

    def __init__(
        self,
        seq_tensor: LongTensor,
        score: float,
        direction: str,
    ) -> None:
        assert direction in {"l2r", "r2l"}
        raw_seq = seq_tensor.tolist()

        if direction == "r2l":
            result = raw_seq[::-1]
        else:
            result = raw_seq

        self.seq = result
        self.score = score

    def __len__(self):
        if len(self.seq) != 0:
            return len(self.seq)
        else:
            return 1

    def __str__(self):
        return f"seq: {self.seq}, score: {self.score}"


class ExpRateRecorder(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)

        self.add_state("total_line", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("rec", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("wrong", default=torch.tensor(0.0), dist_reduce_fx="sum")
        
        with open("./data/vocab.json","r") as f:
            worddict = json.load(f)
        self.int2word = {}
        for k, v in worddict.items():
            self.int2word[v] = k

    def update(self, indices_hat: List[int], indices: List[int]):
        # dist = editdistance.eval(indices_hat, indices)
        self.total_line += 1
        for i,j in zip(indices,indices_hat):
            if self.int2word[i].lower() != self.int2word[j].lower():
                self.wrong += 1
                break

    def compute(self) -> float:
        exp_rate = 1 - self.wrong / self.total_line
        return exp_rate


def loss_sum(output, latex, img, mask):
    return ce_loss(output[0], latex)
    # return ce_loss(output[0],latex) + 0 * img_loss(output[1], img, mask)


def img_loss(preds, gt, mask):
    loss = 0
    loss_fn = F.smooth_l1_loss
    resize_gt = torch.zeros(preds.shape).to(preds.device)
    resize_gt[:, :, :gt.shape[2], :gt.shape[3]] = (gt - torch.min(gt)) / (torch.max(gt) - torch.min(gt))
    pred = (preds - torch.min(preds)) / (torch.max(preds) - torch.min(preds))
    loss = loss_fn(resize_gt.flatten(), pred.flatten())
    return loss


def ce_loss(
    output_hat: torch.Tensor, output: torch.Tensor, ignore_idx: int = vocab.PAD_IDX
) -> torch.Tensor:
    """comput cross-entropy loss

    Args:
        output_hat (torch.Tensor): [batch, len, e]
        output (torch.Tensor): [batch, len]
        ignore_idx (int):

    Returns:
        torch.Tensor: loss value
    """
    flat_hat = rearrange(output_hat, "b l e -> (b l) e")
    flat = rearrange(output, "b l -> (b l)")
    loss = F.cross_entropy(flat_hat, flat, ignore_index=ignore_idx)
    return loss


def latex2img(text):
    """LaTex数学公式转图片

        text        - 文本字符串，其中数学公式须包含在两个$符号之间
        size        - 字号，整型，默认32
        color       - 颜色，浮点型三元组，值域范围[0,1]，默认深黑色
        out         - 文件名，仅支持后缀名为.png的文件名。若维None，则返回PIL图像对象
        kwds        - 关键字参数
                        dpi         - 输出分辨率（每英寸像素数），默认72
                        family      - 系统支持的字体，None表示当前默认的字体
                        weight      - 笔画轻重，可选项包括：normal（默认）、light和bold
        """

    dpi = 72
    family = None
    weight = 'normal'
    size = 32
    color = (0.1, 0.1, 0.1)

    if text[0] + text[-1] != "$$":
        text = "$" + text + "$"

    bfo = BytesIO()  # 创建二进制的类文件对象
    prop = mfm.FontProperties(family=family, size=size, weight=weight)
    mathtext.math_to_image(text, bfo, prop=prop, dpi=dpi)
    im = Image.open(bfo)
    return im


def to_tgt_output(
    tokens: List[List[int]], direction: str, device: torch.device
) -> Tuple[LongTensor, LongTensor]:
    """Generate tgt and out for indices

    Parameters
    ----------
    tokens : List[List[int]]
        indices: [b, l]
    direction : str
        one of "l2f" and "r2l"
    device : torch.device

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        tgt, out: [b, l], [b, l]
    """
    assert direction in {"l2r", "r2l"}

    tokens = [torch.tensor(t, dtype=torch.long) for t in tokens]
    if direction == "l2r":
        tokens = tokens
        start_w = vocab.SOS_IDX
        stop_w = vocab.EOS_IDX
    else:
        tokens = [torch.flip(t, dims=[0]) for t in tokens]
        start_w = vocab.EOS_IDX
        stop_w = vocab.SOS_IDX

    batch_size = len(tokens)
    lens = [len(t) for t in tokens]
    tgt = torch.full(
        (batch_size, max(lens) + 1),
        fill_value=vocab.PAD_IDX,
        dtype=torch.long,
        device=device,
    )
    out = torch.full(
        (batch_size, max(lens) + 1),
        fill_value=vocab.PAD_IDX,
        dtype=torch.long,
        device=device,
    )

    for i, token in enumerate(tokens):
        tgt[i, 0] = start_w
        tgt[i, 1: (1 + lens[i])] = token

        out[i, : lens[i]] = token
        out[i, lens[i]] = stop_w

    return tgt, out


def to_bi_tgt_out(
    tokens: List[List[int]], device: torch.device
) -> Tuple[LongTensor, LongTensor]:
    """Generate bidirection tgt and out

    Parameters
    ----------
    tokens : List[List[int]]
        indices: [b, l]
    device : torch.device

    Returns
    -------
    Tuple[LongTensor, LongTensor]
        tgt, out: [2b, l], [2b, l]
    """

    l2r_tgt, l2r_out = to_tgt_output(tokens, "l2r", device)
    r2l_tgt, r2l_out = to_tgt_output(tokens, "r2l", device)

    tgt = torch.cat((l2r_tgt, r2l_tgt), dim=0)
    out = torch.cat((l2r_out, r2l_out), dim=0)
    return tgt, out


def to_tgt_input(tokens: List[List[int]], ratio: float, device: torch.device, is_test=False) -> Tuple[LongTensor, LongTensor]:
    """Generate tgt and out for indices

    Parameters
    ----------
    tokens : List[List[int]]
        indices: [b, l]
    ratio : float
        ratio to remain token
    device : torch.device

    Returns
    -------
    Tupletorch.Tensor
        tgt: [b, l]
    """
    batch_size = len(tokens)
    subtokens = [[] for t in tokens]    
    for i, token in enumerate(tokens):
        for t in token:
            if random.random() < ratio:
                subtokens[i].append(t)
            else:
                subtokens[i].append(0)
        while len(subtokens[i]) < 30:
            subtokens[i].append(0)
    subtokens = [torch.tensor(t, dtype=torch.long) for t in subtokens]
    
    lens = [len(t) for t in subtokens]
    tgt = torch.full(
        (batch_size, max(lens) + 1),
        fill_value=vocab.PAD_IDX,
        dtype=torch.long,
        device=device,
    )
    for i, token in enumerate(subtokens):
        tgt[i, 0] = vocab.SOS_IDX
        tgt[i, 1:(1+ lens[i])] = token
    
    return tgt