from typing import List

import pytorch_lightning as pl
import torch
import random
from torch import FloatTensor, LongTensor

from bttr.utils import Hypothesis
from einops.einops import rearrange

from .decoder import Decoder
from .encoder import Encoder
from .encoderT import EncoderT
from .labelimg import Up

class BTTR(pl.LightningModule):
    def __init__(
        self,
        d_model: int,
        growth_rate: int,
        num_layers: int,
        nhead: int,
        num_decoder_layers: int,
        dim_feedforward: int,
        dropout: float,
    ):
        super().__init__()

        self.encoder = Encoder(
            d_model=d_model, growth_rate=growth_rate, num_layers=num_layers
        )
        self.encoderT = EncoderT(d_model=d_model, nhead=nhead, d_hid=dim_feedforward, nlayers=num_layers)
        self.decoder = Decoder(
            d_model=d_model,
            nhead=nhead,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
        )
        self.up = Up(in_channels=d_model)

    def forward(
        self, img: FloatTensor, img_mask: LongTensor, tgt: LongTensor, outinput: LongTensor
    ) -> FloatTensor:
        """run img and bi-tgt

        Parameters
        ----------
        img : FloatTensor
            [b, 1, h, w]
        img_mask: LongTensor
            [b, h, w]
        tgt : LongTensor
            [2b, l]

        Returns
        -------
        FloatTensor
            [2b, l, vocab_size]
        """
        feature1, mask = self.encoder(img, img_mask)  # [b, t, d]
        feature = rearrange(feature1, "b h w d -> b (h w) d")
        mask = rearrange(mask, "b h w -> b (h w)")
        # # 双方向，匹配tgt        
        feature = torch.cat((feature, feature), dim=0)  # [2b, t, d]s
        mask = torch.cat((mask, mask), dim=0)
        # 文本信息，注意由于基于tgt encode, tgt 本身已经经过l2r + r2l， size = [2b,l,d]
        featureT = self.encoderT(outinput)
        maskT = (outinput == 0)
        featureT = torch.cat((featureT, featureT), dim=0)  # [2b, t, d]
        maskT = torch.cat((maskT, maskT), dim=0)
        # 文本与图片encode结合
        feature = torch.cat((feature, featureT), dim=1)
        mask = torch.cat((mask, maskT), dim=1)
    
        out = self.decoder(feature, mask, tgt)
        feature2 = rearrange(feature1,"b h w d -> b d h w")
        reimg = self.up(feature2)
        # img = self.GetImage(feature, mask, tgt)
        return out, reimg

    def beam_search(
        self, img: FloatTensor, img_mask: LongTensor, beam_size: int, max_len: int
    ) -> List[Hypothesis]:
        """run bi-direction beam search for given img

        Parameters
        ----------
        img : FloatTensor
            [1, 1, h', w']
        img_mask: LongTensor
            [1, h', w']
        beam_size : int
        max_len : int

        Returns
        -------
        List[Hypothesis]
        """
        feature, mask = self.encoder(img, img_mask)  # [1, t, d]
        feature = rearrange(feature, "b h w d -> b (h w) d")
        mask = rearrange(mask, "b h w -> b (h w)")
        return self.decoder.beam_search(feature, mask, beam_size, max_len)
