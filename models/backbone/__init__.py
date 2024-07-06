from typing import Dict, List

import torch
from torch import nn

from util.misc import NestedTensor
from ..position_encoding import build_position_encoding
from .backbone import *


class Joiner(nn.Sequential):
    def __init__(self, backbone, position_embedding):
        super().__init__(backbone, position_embedding)
        self._export = False

    def forward(self, tensor_list: NestedTensor):
        """
        """
        x = self[0](tensor_list)
        pos = []
        for x_ in x:
            pos.append(self[1](x_, align_dim_orders=False).to(x_.tensors.dtype))
        return x, pos

    def export(self):
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export
    
    def forward_export(self, inputs:torch.Tensor):
        feats, masks = self[0](inputs)
        poss = []
        for feat, mask in zip(feats, masks):
            poss.append(self[1](mask, align_dim_orders=False).to(feat.dtype))
        return feats, None, poss


def build_backbone(args):
    """
    Useful args:
        - encoder: encoder name
        - lr_encoder:
        - dilation
        - use_checkpoint: for swin only for now

    """
    position_embedding = build_position_encoding(args)

    if args.encoder in ['vit_tiny', 'vit_small', 'vit_base', 'res18vd', 'res50vd', 'mobilenetv3_large_1.0']:
        
        backbone = Backbone(
            args.encoder,
            args.vit_encoder_num_layers,
            args.pretrained_encoder,
            window_block_indexes=args.window_block_indexes,
            drop_path=args.drop_path,
            out_channels=args.hidden_dim,
            out_feature_indexes=args.out_feature_indexes,
            projector_scale=args.projector_scale,
        )

    model = Joiner(backbone, position_embedding)
    return model
