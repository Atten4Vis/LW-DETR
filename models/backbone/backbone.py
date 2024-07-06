# ------------------------------------------------------------------------
# LW-DETR
# Copyright (c) 2024 Baidu. All Rights Reserved.
# Licensed under the Apache License, Version 2.0 [see LICENSE for details]
# ------------------------------------------------------------------------
# Modified from Conditional DETR (https://github.com/Atten4Vis/ConditionalDETR)
# Copyright (c) 2021 Microsoft. All Rights Reserved.
# ------------------------------------------------------------------------
# Copied from DETR (https://github.com/facebookresearch/detr)
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# ------------------------------------------------------------------------

"""
Backbone modules.
"""
from functools import partial
import torch
import torch.nn.functional as F
from torch import nn

from util.misc import NestedTensor, is_main_process

from .base import BackboneBase
from .vit import ViT
from .presnet import PResNet
from .projector import MultiScaleProjector

__all__ = ["Backbone"]


class Backbone(BackboneBase):
    """backbone."""
    def __init__(self,
                 name: str,
                 vit_encoder_num_layers: int,
                 pretrained_encoder: str=None,
                 window_block_indexes: list=None,
                 drop_path=0.0,
                 out_channels=256,
                 out_feature_indexes: list=None,
                 projector_scale: list=None,
                 ):
        super().__init__()
        self.name = name
        if 'vit' in name:
            if name == 'vit_tiny':
                img_size, embed_dim, depth, num_heads, dp = 1024, 192, 12, 12, 0.
            elif name == 'vit_small':
                img_size, embed_dim, depth, num_heads, dp = 1024, 384, 12, 12, 0.
            elif name == 'vit_base':
                img_size, embed_dim, depth, num_heads, dp = 1024, 768, 12, 12, 0.
            else:
                raise NotImplementedError("Backbone {} is not support now.".format(name))

            depth = vit_encoder_num_layers
            assert window_block_indexes is not None

            dp = drop_path

            self.encoder = ViT(  # Single-scale ViT encoder
                img_size=img_size,
                patch_size=16,
                embed_dim=embed_dim,
                depth=depth,
                num_heads=num_heads,
                drop_path_rate=dp,
                mlp_ratio=4,
                qkv_bias=True,
                norm_layer=partial(nn.LayerNorm, eps=1e-6),
                window_block_indexes=window_block_indexes,
                use_act_checkpoint=False,  # use checkpoint to save memory
                use_abs_pos=True,
                out_feature_indexes=out_feature_indexes,
                use_cae=True,
            )
            # load pretrain model in main_process
            if pretrained_encoder is not None:
                if is_main_process() and pretrained_encoder is not None:
                    checkpoint = torch.load(pretrained_encoder, map_location="cpu")
                    checkpoint_dict = {
                        k.replace('encoder.', ""): v 
                        for k, v in checkpoint["model"].items()
                    }
                    stat = self.encoder.load_state_dict(checkpoint_dict, strict=False)
                    print(stat)
        elif 'res' in name:
            if name == 'res18vd':
                depth = 18
                freeze_at = -1
                return_idx = [1, 2, 3]
                freeze_norm = False
            elif name == 'res50vd':
                depth = 50
                freeze_at = 0
                return_idx = [1, 2, 3]
                freeze_norm = True
            else:
                raise NotImplementedError("Backbone {} is not support now.".format(name))

            self.encoder = PResNet(
                depth=depth, 
                variant='d', 
                num_stages=4, 
                return_idx=return_idx, 
                act='relu',
                freeze_at=freeze_at, 
                freeze_norm=freeze_norm
            )

            # load pretrain model in main_process
            if pretrained_encoder is not None:
                if is_main_process() and pretrained_encoder is not None:
                    checkpoint = torch.load(pretrained_encoder, map_location="cpu")
                    stat = self.encoder.load_state_dict(checkpoint, strict=False)
                    print(stat)
        else:
            raise NotImplementedError("Backbone {} is not support now.".format(name))

        # build encoder + projector as backbone module

        self.projector_scale = projector_scale
        assert len(self.projector_scale) > 0
        assert sorted(self.projector_scale) == self.projector_scale, \
            "only support projector scale P3/P4/P5/P6 in ascending order."
        level2scalefactor = dict(
            P3=2.0,
            P4=1.0,
            P5=0.5,
            P6=0.25
        )
        scale_factors = [level2scalefactor[lvl] for lvl in self.projector_scale]

        self.projector = MultiScaleProjector(
            in_channels=self.encoder._out_feature_channels,
            out_channels=out_channels,
            scale_factors=scale_factors,
        )

        self._export = False

    def export(self):
        self._export = True
        self._forward_origin = self.forward
        self.forward = self.forward_export

    def forward(self, tensor_list: NestedTensor):
        """
        """
        # (H, W, B, C)
        feats = self.encoder(tensor_list.tensors)
        feats = self.projector(feats)
        # x: [(B, C, H, W)]
        out = []
        for feat in feats:
            m = tensor_list.mask
            assert m is not None
            mask = F.interpolate(m[None].float(), size=feat.shape[-2:]).to(torch.bool)[0]
            out.append(NestedTensor(feat, mask))
        return out

    def forward_export(self, tensors:torch.Tensor):
        feats = self.encoder(tensors)
        feats = self.projector(feats)
        out_feats = []
        out_masks = []
        for feat in feats:
            # x: [(B, C, H, W)]
            b, _, h, w = feat.shape
            out_masks.append(torch.zeros((b, h, w), dtype=torch.bool, device=feat.device))
            out_feats.append(feat)
        return out_feats, out_masks

    def get_named_param_lr_pairs(self, args, prefix:str = "backbone.0"):
        if 'vit' in self.name:
            num_layers = args.vit_encoder_num_layers
            backbone_key = 'backbone.0.encoder'
            named_param_lr_pairs = {}
            for n, p in self.named_parameters():
                n = prefix + "." + n
                if backbone_key in n and p.requires_grad:
                    lr = args.lr_encoder * get_vit_lr_decay_rate(
                        n, lr_decay_rate=args.lr_vit_layer_decay, 
                        num_layers=num_layers) * args.lr_component_decay ** 2
                    wd = args.weight_decay * get_vit_weight_decay_rate(n)
                    named_param_lr_pairs[n] = {
                        "params": p,
                        "lr": lr,
                        "weight_decay": wd
                    }
        elif 'res' in self.name:
            backbone_key = 'backbone.0.encoder'
            named_param_lr_pairs = {}
            for n, p in self.named_parameters():
                n = prefix + "." + n
                if backbone_key in n and p.requires_grad:
                    lr = 0.1 * args.lr
                    wd = args.weight_decay * get_vit_weight_decay_rate(n)
                    named_param_lr_pairs[n] = {
                        "params": p,
                        "lr": lr,
                        "weight_decay": wd
                    }
        else:
            raise NotImplementedError
        return named_param_lr_pairs


def get_vit_lr_decay_rate(name, lr_decay_rate=1.0, num_layers=12):
    """
    Calculate lr decay rate for different ViT blocks.

    Args:
        name (string): parameter name.
        lr_decay_rate (float): base lr decay rate.
        num_layers (int): number of ViT blocks.
    Returns:
        lr decay rate for the given parameter.
    """
    layer_id = num_layers + 1
    if name.startswith("backbone"):
        if ".pos_embed" in name or ".patch_embed" in name:
            layer_id = 0
        elif ".blocks." in name and ".residual." not in name:
            layer_id = int(name[name.find(".blocks.") :].split(".")[2]) + 1
    print("name: {}, lr_decay: {}".format(name, lr_decay_rate ** (num_layers + 1 - layer_id)))
    return lr_decay_rate ** (num_layers + 1 - layer_id)


def get_vit_weight_decay_rate(name, weight_decay_rate=1.0):
    if ('gamma' in name) or ('pos_embed' in name) or ('rel_pos' in name) or ('bias' in name) or ('norm' in name):
        weight_decay_rate = 0.
    print("name: {}, weight_decay rate: {}".format(name, weight_decay_rate))
    return weight_decay_rate
