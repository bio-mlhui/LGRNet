from typing import Optional

import torch
from torch import nn, Tensor
from torch.nn.functional import pad
from torch.nn.init import trunc_normal_

from natten.functional import na2d_av, na2d_qk_with_bias
from einops import rearrange
from natten import NeighborhoodAttention2D
from detectron2.modeling import META_ARCH_REGISTRY

class NeighborhoodAttention2D_qk(nn.Module):
    """
    Neighborhood Attention 2D Module
    """

    def __init__(
        self,
        dim: int,
        num_heads: int,
        kernel_size: int,
        dilation: int = 1,
        bias: bool = True,
        qkv_bias: bool = True,
        qk_scale: Optional[float] = None,
        attn_drop: float = 0.0,
        proj_drop: float = 0.0,
    ):
        super().__init__()
        self.num_heads = num_heads
        self.head_dim = dim // self.num_heads
        self.scale = qk_scale or self.head_dim**-0.5
        assert (
            kernel_size > 1 and kernel_size % 2 == 1
        ), f"Kernel size must be an odd number greater than 1, got {kernel_size}."
        self.kernel_size = kernel_size
        assert (
            dilation is None or dilation >= 1
        ), f"Dilation must be greater than or equal to 1, got {dilation}."
        self.dilation = dilation or 1
        self.window_size = self.kernel_size * self.dilation

        self.q_linear = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv_linear = nn.Linear(dim, dim * 2, bias=qkv_bias)
        if bias:
            self.rpb = nn.Parameter(
                torch.zeros(num_heads, (2 * kernel_size - 1), (2 * kernel_size - 1))
            )
            trunc_normal_(self.rpb, std=0.02, mean=0.0, a=-2.0, b=2.0)
        else:
            self.register_parameter("rpb", None)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, 
                x_q: Tensor, 
                x_k: Tensor) -> Tensor:
        # bt h w c; bt h w c, 前一帧
        if x_q.dim() != 4:
            raise ValueError(
                f"NeighborhoodAttention2D expected a rank-4 input tensor; got {x.dim()=}."
            )

        B, H, W, C = x_q.shape
        # Pad if the input is small than the minimum supported size
        H_padded, W_padded = H, W
        padding_h = padding_w = 0
        if H < self.window_size or W < self.window_size:
            padding_h = max(0, self.window_size - H_padded)
            padding_w = max(0, self.window_size - W_padded)
            x_q = pad(x_q, (0, 0, 0, padding_w, 0, padding_h))
            x_k = pad(x_k, (0, 0, 0, padding_w, 0, padding_h))
            _, H_padded, W_padded, _ = x_q.shape

        # b h w c -> b h w h c_h
        q = self.q_linear(x_q).reshape(B, H_padded, W_padded, self.num_heads, self.head_dim)
        q = q.permute(0, 3, 1, 2, 4) # b head h w c_h
        kv = (
            self.kv_linear(x_k)
            .reshape(B, H_padded, W_padded, 2, self.num_heads, self.head_dim)
            .permute(3, 0, 4, 1, 2, 5)
        ) # b 
        k, v = kv[0], kv[1]
        q = q * self.scale
        attn = na2d_qk_with_bias(q, k, self.rpb, self.kernel_size, self.dilation)
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        x_q = na2d_av(attn, v, self.kernel_size, self.dilation) # b head h w c_h
        x_q = x_q.permute(0, 2, 3, 1, 4).reshape(B, H_padded, W_padded, C) # b h w head c_h

        # Remove padding, if added any
        if padding_h or padding_w:
            x_q = x_q[:, :H, :W, :].contiguous()

        return self.proj_drop(self.proj(x_q))

    def extra_repr(self) -> str:
        return (
            f"head_dim={self.head_dim}, num_heads={self.num_heads}, "
            + f"kernel_size={self.kernel_size}, dilation={self.dilation}, "
            + f"has_bias={self.rpb is not None}"
        )

from models.layers.utils import _get_clones
class NA_qk_Layer(nn.Module):

    def __init__(self, d_model, configs):
        super().__init__()
        self.self_attn = NeighborhoodAttention2D_qk(dim=configs['d_model'],
                                                    num_heads=configs['num_heads'],
                                                    kernel_size=configs['kernel_size'],
                                                    dilation=configs['dilation'],
                                                    bias=False,
                                                    qkv_bias=False,)
        self.num_steps = configs['num_steps'] if 'num_steps' in configs else 1
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(configs['dropout'])
    

    def forward(self, tgt=None, scale_shapes=None, level_start_idxs=None, nf=None):
        # bt hw_sigma c -> list[b t h w c], 3
        video_feats = [tgt[:, start_idx:(start_idx + haosen[0]*haosen[1])].contiguous() for start_idx, haosen in zip(level_start_idxs, scale_shapes)]
        video_feats = [rearrange(haosen, '(b t) (h w) c -> b t h w c', t=nf, h=scale_shapes[idx][0], w=scale_shapes[idx][1]).contiguous() for idx, haosen in enumerate(video_feats)]

        video_key_feats = []
        for haosen in video_feats:
            scale_feats = torch.stack([torch.roll(haosen, shifts=k, dims=1) for k in range(1, self.num_steps+1)], dim=0) # s b t h w c
            video_key_feats.append(scale_feats.flatten(0, 2)) #sbt h w c
        
        # sbt h w c
        video_feats = [haosen.unsqueeze(0).repeat(self.num_steps, 1,1,1,1,1).flatten(0, 2) for haosen in video_feats]
 
        local_feats = [] # list[sbt h w c]
        for idx, (q_feat, k_feat) in enumerate(zip(video_feats, video_key_feats)):
            local_feats.append(self.self_attn(q_feat, k_feat))
        local_feats = [rearrange(haosen, '(s bt) h w c -> s bt h w c',s=self.num_steps) for haosen in local_feats]
        local_feats = [haosen.sum(dim=0) for haosen in local_feats] # bt h w c
        local_feats = torch.cat([haosen.flatten(1, 2) for haosen in local_feats], dim=1) # bt hw_sigma c
        tgt = tgt + self.dropout(local_feats)
        tgt = self.norm(tgt) 
        return tgt

@META_ARCH_REGISTRY.register()
class NA_qk_Layer_v2(nn.Module):

    def __init__(self, configs):
        super().__init__()
        self.self_attn = NeighborhoodAttention2D_qk(dim=configs['d_model'],
                                                    num_heads=configs['num_heads'],
                                                    kernel_size=configs['kernel_size'],
                                                    dilation=configs['dilation'],
                                                    bias=False,
                                                    qkv_bias=False,)
    

    def forward(self,
                query=None, 
                spatial_shapes=None,
                level_start_index=None,
                video_aux_dict=None,):
        
        # bt hw_sigma c -> list[b t h w c], 3
        video_feat = [query[:, start_idx:(start_idx + haosen[0]*haosen[1])].contiguous() for start_idx, haosen in zip(level_start_index, spatial_shapes)]
        video_feat = [rearrange(haosen, '(b t) (h w) c -> b t h w c',t=video_aux_dict['nf'], h=spatial_shapes[idx][0], w=spatial_shapes[idx][1]).contiguous() for idx, haosen in enumerate(video_feat)]
        video_key_feats = [torch.roll(haosen, shifts=1, dims=1).contiguous() for haosen in video_feat]

        local_feats = [] # list[bt h w c]
        for idx, (q_feat, k_feat) in enumerate(zip(video_feat, video_key_feats)):
            local_feats.append(self.self_attn(q_feat.flatten(0, 1), k_feat.flatten(0, 1)))
        
        local_feats = torch.cat([haosen.flatten(1, 2) for haosen in local_feats], dim=1) # bt hw_sigma c
        return local_feats, None
