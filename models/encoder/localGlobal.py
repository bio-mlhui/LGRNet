# Copyright (c) Facebook, Inc. and its affiliates.
import logging
import numpy as np
from typing import Callable, Dict, List, Optional, Tuple, Union

import fvcore.nn.weight_init as weight_init
import torch
from torch import nn
from torch.nn import functional as F
from torch.nn.init import xavier_uniform_, constant_, uniform_, normal_
from torch.cuda.amp import autocast

from detectron2.config import configurable
from detectron2.layers import Conv2d, ShapeSpec, get_norm
from detectron2.modeling import META_ARCH_REGISTRY
from models.layers.position_encoding import PositionEmbeddingSine
from models.layers.utils import _get_clones, _get_activation_fn
from .ops.modules import MSDeformAttn

# MSDeformAttn Transformer encoder in deformable detr
class MSDeformAttnTransformerEncoderOnly(nn.Module):
    def __init__(self, d_model=256, nhead=8,
                 num_encoder_layers=6, dim_feedforward=1024, dropout=0.1,
                 activation="relu",
                 num_feature_levels=4, enc_n_points=4,
                 add_local = False,
                 add_global=False,
                 local_configs=None,
                 global_configs=None,
                 frame_nqueries=None,
        ):
        super().__init__()

        self.d_model = d_model
        self.nhead = nhead

        encoder_layer = MSDeformAttnTransformerEncoderLayer(d_model = d_model,
                                                            d_ffn = dim_feedforward,
                                                            dropout = dropout,
                                                            activation = activation,
                                                            n_levels = num_feature_levels,
                                                            n_heads = nhead,
                                                            n_points = enc_n_points,
                                                            add_local = add_local,
                                                            add_global = add_global,
                                                            local_configs = local_configs,
                                                            global_configs = global_configs
                                                            )
        self.encoder = MSDeformAttnTransformerEncoder(encoder_layer, num_encoder_layers,
                                                      d_model=d_model,
                                                      frame_nqueries=frame_nqueries)

        self.level_embed = nn.Parameter(torch.Tensor(num_feature_levels, d_model))

        self._reset_parameters()

    def _reset_parameters(self):
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        for m in self.modules():
            if isinstance(m, MSDeformAttn):
                m._reset_parameters()
        normal_(self.level_embed)

    def get_valid_ratio(self, mask):
        _, H, W = mask.shape # b h w
        valid_H = torch.sum(~mask[:, :, 0], 1) # b 
        valid_W = torch.sum(~mask[:, 0, :], 1) # b
        valid_ratio_h = valid_H.float() / H 
        valid_ratio_w = valid_W.float() / W
        valid_ratio = torch.stack([valid_ratio_w, valid_ratio_h], -1) # b 2
        return valid_ratio

    def forward(self, 
                srcs=None, 
                pos_embeds=None,
                video_aux_dict=None,
                **kwargs):
        masks = [torch.zeros((x.size(0), x.size(2), x.size(3)), device=x.device, dtype=torch.bool) for x in srcs]
        # prepare input for encoder
        src_flatten = []
        mask_flatten = []
        lvl_pos_embed_flatten = []
        spatial_shapes = []
        for lvl, (src, mask, pos_embed) in enumerate(zip(srcs, masks, pos_embeds)):
            bs, c, h, w = src.shape
            spatial_shape = (h, w)
            spatial_shapes.append(spatial_shape)
            src = src.flatten(2).transpose(1, 2)
            mask = mask.flatten(1)
            pos_embed = pos_embed.flatten(2).transpose(1, 2)
            lvl_pos_embed = pos_embed + self.level_embed[lvl].view(1, 1, -1)
            lvl_pos_embed_flatten.append(lvl_pos_embed)
            src_flatten.append(src)
            mask_flatten.append(mask)
        src_flatten = torch.cat(src_flatten, 1)
        mask_flatten = torch.cat(mask_flatten, 1)
        lvl_pos_embed_flatten = torch.cat(lvl_pos_embed_flatten, 1)
        spatial_shapes = torch.as_tensor(spatial_shapes, dtype=torch.long, device=src_flatten.device)
        level_start_index = torch.cat((spatial_shapes.new_zeros((1, )), spatial_shapes.prod(1).cumsum(0)[:-1]))
        valid_ratios = torch.stack([self.get_valid_ratio(m) for m in masks], 1) # b #scale 2

        # encoder
        memory, frame_feats, frame_poses = self.encoder(src=src_flatten,  # bt hw_sigma c
                                                        spatial_shapes=spatial_shapes, 
                                                        level_start_index=level_start_index, 
                                                        valid_ratios=valid_ratios, 
                                                        pos=lvl_pos_embed_flatten, 
                                                        padding_mask=mask_flatten,
                                                        video_aux_dict=video_aux_dict)

        return memory, spatial_shapes, level_start_index, frame_feats, frame_poses

class MSDeformAttnTransformerEncoderLayer(nn.Module):
    def __init__(self,
                 d_model=256, d_ffn=1024,
                 dropout=0.1, activation="relu",
                 n_levels=4, n_heads=8, n_points=4,
                 add_local=False,
                 add_global=False,
                 local_configs=None,
                 global_configs=None):
        super().__init__()
        # deform2d
        self.self_attn = MSDeformAttn(d_model, n_levels, n_heads, n_points)
        self.dropout1 = nn.Dropout(dropout)
        self.norm1 = nn.LayerNorm(d_model)

        self.add_local = add_local
        if self.add_local:
            from .neighborhood_qk import NA_qk_Layer
            # self
            self.local_cnp = NA_qk_Layer(d_model=d_model, configs=local_configs)


        self.add_global = add_global
        if self.add_global:
            from models.layers.decoder_layers import CrossAttentionLayer 
            # cross
            self.frame_query_cross_multiscale = CrossAttentionLayer(d_model=d_model, nhead=8, dropout=0.0,
                                                                    activation="relu", normalize_before=False)
            self.cross_num_heads = 8
            self.global_add_attn_mask = global_configs['add_attn_mask'] if 'add_attn_mask' in global_configs else False
            # self+ffn
            from models.encoder.ops.modules.frame_query_ss2d import FrameQuery_SS2DLayer_hilbert 
            self.global_hiss = FrameQuery_SS2DLayer_hilbert(global_configs)

            self.multiscale_cross_query = CrossAttentionLayer(d_model=d_model, nhead=8, dropout=0.0,
                                                              activation="relu", normalize_before=False)
        # ffn
        self.linear1 = nn.Linear(d_model, d_ffn)
        self.activation = _get_activation_fn(activation)
        self.dropout2 = nn.Dropout(dropout)
        self.linear2 = nn.Linear(d_ffn, d_model)
        self.dropout3 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)

    @staticmethod
    def with_pos_embed(tensor, pos):
        return tensor if pos is None else tensor + pos

    def forward_ffn(self, src):
        src2 = self.linear2(self.dropout2(self.activation(self.linear1(src))))
        src = src + self.dropout3(src2)
        src = self.norm2(src)
        return src

    @torch.no_grad()
    def get_attn_mask(self, frame_query_feats, src, spatial_shapes, level_start_index,):
        # nq bt c
        # bt hw_sigma c
        assert len(spatial_shapes) == 3
        frame_query_feats = frame_query_feats.permute(1, 0, 2) # bt nq c
        feat = src[:, level_start_index[-1]: (level_start_index[-1] + spatial_shapes[-1][0] * spatial_shapes[-1][1])]
        feat = rearrange(feat, 'b (h w) c -> b c h w',h=spatial_shapes[-1][0],w=spatial_shapes[-1][1])
        mask = torch.einsum('bnc, bchw -> b n h w',frame_query_feats, feat)
        mask_2 = F.interpolate(mask, size=spatial_shapes[0].tolist(), mode='bilinear',align_corners=False)
        mask_3 = F.interpolate(mask, size=spatial_shapes[1].tolist(), mode='bilinear', align_corners=False)
        attn_mask = torch.cat([mask_2.flatten(2), mask_3.flatten(2), mask.flatten(2)], dim=-1) #bt n hw_sigma
        attn_mask = (attn_mask.unsqueeze(1).repeat(1, self.cross_num_heads, 1, 1).flatten(0, 1).sigmoid() < 0.5).bool()
        return attn_mask

    
    def forward(self, 
                src=None, pos=None, 
                reference_points=None, spatial_shapes=None, level_start_index=None, padding_mask=None,
                video_aux_dict=None,
                frame_query_feats=None, # nq bt c
                frame_query_poses=None):
        
        if self.add_local:
            # local_self
            src = self.local_cnp(tgt=src, 
                                scale_shapes=spatial_shapes,
                                level_start_idxs=level_start_index,
                                nf=video_aux_dict['nf']) 
        if self.add_global:
            if self.global_add_attn_mask:
                attn_mask = self.get_attn_mask(frame_query_feats, src, spatial_shapes, level_start_index,) # bthead nq hw
                attn_mask[torch.where(attn_mask.sum(-1) == attn_mask.shape[-1])] = False  # 全masked掉的 全注意, 比如有padding
            else:
                attn_mask = None
            # cross
            frame_query_feats = self.frame_query_cross_multiscale(
                tgt=frame_query_feats,  # nq bt c
                memory=src.permute(1, 0, 2), # hw_sigma bt c
                memory_mask=attn_mask,
                memory_key_padding_mask=None,
                pos= pos.permute(1,0,2),
                query_pos=frame_query_poses,
            )
            # self+ffn
            frame_query_feats = self.global_hiss(frame_query_feats=frame_query_feats,
                                                 frame_query_poses=frame_query_poses,
                                                 video_aux_dict=video_aux_dict)

            # self
            src = self.multiscale_cross_query(
                tgt=src.permute(1, 0, 2),  # hw_sigma bt c
                memory=frame_query_feats, # nq bt c
                memory_mask=None,
                memory_key_padding_mask=None,
                pos= frame_query_poses,
                query_pos=pos.permute(1,0,2),                
            ).permute(1, 0, 2)

        # self attention
        src2 = self.self_attn(self.with_pos_embed(src, pos), reference_points, src, spatial_shapes, level_start_index, padding_mask)[0]
        src = src + self.dropout1(src2)
        src = self.norm1(src)

        # ffn
        src = self.forward_ffn(src)
        return src, frame_query_feats


class MSDeformAttnTransformerEncoder(nn.Module):
    def __init__(self,
                encoder_layer=None, 
                num_layers=None, 
                d_model=None, frame_nqueries=None):
        super().__init__()
        self.layers = _get_clones(encoder_layer, num_layers)
        self.num_layers = num_layers

        self.frame_nqueries = frame_nqueries # 10
        self.frame_query_feats = nn.Embedding(self.frame_nqueries, d_model)
        self.frame_query_poses = nn.Embedding(self.frame_nqueries, d_model)

    @staticmethod
    def get_reference_points(spatial_shapes, valid_ratios, device):
        # b #scale 2, valid_w(0-1), valid_h(0-1), 整个feature map有多少是非padding的
        # list[h w] #scale
        reference_points_list = []
        for lvl, (H_, W_) in enumerate(spatial_shapes):
            ref_y, ref_x = torch.meshgrid(torch.linspace(0.5, H_ - 0.5, H_, dtype=torch.float32, device=device),
                                          torch.linspace(0.5, W_ - 0.5, W_, dtype=torch.float32, device=device))
            ref_y = ref_y.reshape(-1)[None] / (valid_ratios[:, None, lvl, 1] * H_) # 1 hw / b 1 -> b hw(0-1), y的绝对坐标
            ref_x = ref_x.reshape(-1)[None] / (valid_ratios[:, None, lvl, 0] * W_) # 1 hw / b 1 -> b hw(0-1), x的绝对坐标
            ref = torch.stack((ref_x, ref_y), -1) # b hw 2
            reference_points_list.append(ref)
        reference_points = torch.cat(reference_points_list, 1) # b hw_sigma 2, 每个点的相对坐标(0-1)
        reference_points = reference_points[:, :, None] * valid_ratios[:, None] # b hw_sigma 1 2 * b 1 #scale 2
        return reference_points # b hw_sigma #scale 2

    def forward(self, 
                src,  # bt hw_sigma c
                spatial_shapes, 
                level_start_index, 
                valid_ratios, 
                pos=None, 
                padding_mask=None,
                video_aux_dict=None):
        
        output = src # bt hw_sigma c
        batch_size_nf = output.shape[0]
        frame_query_feats = self.frame_query_feats.weight.unsqueeze(1).repeat(1,batch_size_nf, 1)
        frame_query_poses = self.frame_query_poses.weight.unsqueeze(1).repeat(1,batch_size_nf,1) # n bt c
        
        reference_points = self.get_reference_points(spatial_shapes, valid_ratios, device=src.device)
        frame_feats = []
        for _, layer in enumerate(self.layers):
            output, frame_query_feats = layer(src=output, 
                                                pos=pos, 
                                                reference_points=reference_points, 
                                                spatial_shapes=spatial_shapes, 
                                                level_start_index=level_start_index, 
                                                padding_mask=padding_mask,
                                                video_aux_dict=video_aux_dict,
                                                frame_query_feats=frame_query_feats,
                                                frame_query_poses=frame_query_poses)
            frame_feats.append(frame_query_feats)

        return output, frame_feats, frame_query_poses


import copy
from einops import rearrange
from models.layers.utils import _get_clones
from models.layers.position_encoding import build_position_encoding
# video multiscale, text_dict

@META_ARCH_REGISTRY.register()
class Video_Deform2D_DividedTemporal_MultiscaleEncoder_localGlobal(nn.Module):
    def __init__(
        self,
        configs,
        multiscale_shapes, # {'res2': .temporal_stride, .spatial_stride, .dim}
    ):
        super().__init__()
        d_model = configs['d_model']
        fpn_norm = configs['fpn_norm'] # fpn的norm
        nlayers = configs['nlayers']

        # 4, 8, 16, 32
        self.multiscale_shapes = dict(sorted(copy.deepcopy(multiscale_shapes).items(), key=lambda x: x[1].spatial_stride))
        self.encoded_scales = sorted(configs['encoded_scales'], 
                                     key=lambda x:self.multiscale_shapes[x].spatial_stride) # res3, res4, res5
        
        # 4 -> 8 -> 16 -> 32    
        self.scale_dims = [val.dim for val in multiscale_shapes.values()]
        self.video_projs = META_ARCH_REGISTRY.get(configs['video_projs']['name'])(configs=configs['video_projs'],
                                                                            multiscale_shapes=multiscale_shapes, out_dim=d_model)

        self.pos_2d = build_position_encoding(position_embedding_name='2d')

        deform_attn = configs['deform_attn']
        self.transformer = MSDeformAttnTransformerEncoderOnly(
            d_model=d_model,
            dropout=deform_attn['dropout'],
            nhead=deform_attn['nheads'],
            dim_feedforward=deform_attn['dim_feedforward'],
            activation=deform_attn['activation'],
            num_encoder_layers=nlayers,
            num_feature_levels=len(self.encoded_scales),
            enc_n_points=deform_attn['enc_n_points'],
            add_local = configs['add_local'],
            add_global = configs['add_global'],
            local_configs = configs['local_configs'],
            global_configs = configs['global_configs'],
            frame_nqueries=configs['frame_nqueries']
                
        )

        min_encode_stride = self.multiscale_shapes[self.encoded_scales[0]].spatial_stride # 8
        min_stride = list(self.multiscale_shapes.values())[0].spatial_stride # 4
        self.num_fpn_levels = int(np.log2(min_encode_stride) - np.log2(min_stride))
        lateral_convs = [] 
        output_convs = []
        use_bias = fpn_norm == ""
        for idx, in_channels in enumerate(self.scale_dims[:self.num_fpn_levels]):
            lateral_norm = get_norm(fpn_norm, d_model)
            output_norm = get_norm(fpn_norm, d_model)

            lateral_conv = Conv2d(in_channels, d_model, kernel_size=1, bias=use_bias, norm=lateral_norm)
            output_conv = Conv2d(d_model, d_model, kernel_size=3, padding=1, bias=use_bias, norm=output_norm, activation=F.relu)

            self.add_module("adapter_{}".format(idx + 1), lateral_conv)
            self.add_module("layer_{}".format(idx + 1), output_conv)

            lateral_convs.append(lateral_conv)
            output_convs.append(output_conv)
        # Place convs into top-down order (from low to high resolution)
        # to make the top-down computation in forward clearer.
        self.lateral_convs = lateral_convs[::-1] # 8 4
        self.output_convs = output_convs[::-1] # 8 4

    def forward(self, 
                multiscales=None, # b c t h w
                video_aux_dict=None, # dict{}
                **kwargs):
        batch_size, _, nf = multiscales[list(multiscales.keys())[0]].shape[:3]
        video_aux_dict['nf'] = nf
        multiscales = self.video_projs(multiscales) 
        assert set(list(multiscales.keys())).issubset(set(list(self.multiscale_shapes.keys())))
        assert set(list(self.multiscale_shapes.keys())).issubset(set(list(multiscales.keys())))

        srcs = []
        poses = [] # 32, 16, 8
        for idx, scale_name in enumerate(self.encoded_scales[::-1]):
            x = multiscales[scale_name].permute(0, 2, 1, 3, 4).flatten(0,1).contiguous() # bt c h w
            srcs.append(x)
            poses.append(self.pos_2d(torch.zeros_like(x)[:, 0, :, :].bool(), hidden_dim=x.shape[1]))

        memory, spatial_shapes, level_start_index, frame_feats, frame_poses = self.transformer(srcs=srcs, 
                                                                                                pos_embeds=poses,
                                                                                                video_aux_dict=video_aux_dict)
        bs = memory.shape[0]
        spatial_index = 0
        memory_features = [] # 32 16 8
        for lvl in range(len(self.encoded_scales)):
            h, w = spatial_shapes[lvl]
            memory_lvl = memory[:, spatial_index : spatial_index + h * w, :].reshape(bs, h, w, -1).permute(0, 3, 1, 2).contiguous()  
            memory_features.append(memory_lvl)
            spatial_index += h * w

        for idx, f in enumerate(list(self.multiscale_shapes.keys())[:self.num_fpn_levels][::-1]):
            x = multiscales[f].permute(0, 2, 1, 3, 4).flatten(0,1).contiguous() # bt c h w
            cur_fpn = self.lateral_convs[idx](x)
            y = cur_fpn + F.interpolate(memory_features[-1], size=cur_fpn.shape[-2:], mode="bilinear", align_corners=False)
            y = self.output_convs[idx](y)
            memory_features.append(y)

        assert len(memory_features) == len(list(self.multiscale_shapes.keys()))

        ret = {}
        for key, out_feat in zip(list(self.multiscale_shapes.keys()), memory_features[::-1]):
            ret[key] = rearrange(out_feat, '(b t) c h w -> b c t h w', b=batch_size, t=nf)
        return ret, frame_feats[::-1], frame_poses # 32, 16, 8

