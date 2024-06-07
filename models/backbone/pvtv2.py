import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath
from timm.models.registry import register_model

class DWConv(nn.Module):
    def __init__(self, dim):
        super(DWConv, self).__init__()
        self.dwconv = nn.Conv2d(dim, dim, 3, 1, 1, groups=dim)

    def forward(self, x, H, W):
        B,N,C = x.shape
        x     = x.transpose(1, 2).view(B, C, H, W)
        x     = self.dwconv(x)
        x     = x.flatten(2).transpose(1, 2)
        return x

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features):
        super().__init__()
        self.fc1        = nn.Linear(in_features, hidden_features)
        self.dwconv     = DWConv(hidden_features)
        self.fc2        = nn.Linear(hidden_features, in_features)

    def forward(self, x, H, W):
        x = self.fc1(x)
        x = F.gelu(self.dwconv(x, H, W))
        x = self.fc2(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads, sr_ratio):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."

        self.num_heads = num_heads
        self.scale     = (dim//num_heads)**(-0.5)
        self.q         = nn.Linear(dim, dim)
        self.kv        = nn.Linear(dim, dim*2)
        self.proj      = nn.Linear(dim, dim)
        self.sr_ratio  = sr_ratio
        if sr_ratio > 1:
            self.sr    = nn.Conv2d(dim, dim, kernel_size=sr_ratio, stride=sr_ratio)
            self.norm  = nn.LayerNorm(dim)

    def forward(self, x, H, W):
        B, N, C = x.shape
        q       = self.q(x).reshape(B, N, self.num_heads, C//self.num_heads).permute(0, 2, 1, 3)

        if self.sr_ratio > 1:
            x_ = x.permute(0, 2, 1).reshape(B, C, H, W)
            x_ = self.sr(x_).reshape(B, C, -1).permute(0, 2, 1)
            x_ = self.norm(x_)
            kv = self.kv(x_).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        else:
            kv = self.kv(x).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
 
        k, v = kv[0], kv[1]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        x    = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x    = self.proj(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio, drop_path, sr_ratio):
        super().__init__()
        self.norm1     = nn.LayerNorm(dim, eps=1e-6)
        self.attn      = Attention(dim, num_heads=num_heads, sr_ratio=sr_ratio)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2     = nn.LayerNorm(dim, eps=1e-6)
        self.mlp       = Mlp(in_features=dim, hidden_features=int(dim*mlp_ratio))

    def forward(self, x, H, W):
        x = x + self.drop_path(self.attn(self.norm1(x), H, W))
        x = x + self.drop_path(self.mlp(self.norm2(x), H, W))
        return x

class OverlapPatchEmbed(nn.Module):
    def __init__(self, patch_size, stride, in_chans, embed_dim):
        super().__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=stride, padding=(patch_size//2, patch_size//2))
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, x):
        x       = self.proj(x)
        B,C,H,W = x.shape
        x       = x.flatten(2).transpose(1, 2)
        x       = self.norm(x)
        return x, H, W

class PVT(nn.Module):
    def __init__(self, embed_dims, mlp_ratios, depths, snapshot, sr_ratios=[8, 4, 2, 1]):
        super().__init__()
        self.depths       = depths
        self.snapshot     = snapshot
        # patch_embed
        self.patch_embed1 = OverlapPatchEmbed(patch_size=7, stride=4, in_chans=3,             embed_dim=embed_dims[0])
        self.patch_embed2 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[0], embed_dim=embed_dims[1])
        self.patch_embed3 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[1], embed_dim=embed_dims[2])
        self.patch_embed4 = OverlapPatchEmbed(patch_size=3, stride=2, in_chans=embed_dims[2], embed_dim=embed_dims[3])
        # transformer encoder
        dpr = [x.item() for x in torch.linspace(0, 0.1, sum(depths))]  # stochastic depth decay rule
        cur = 0
        self.block1 = nn.ModuleList([Block(dim=embed_dims[0], num_heads=1, mlp_ratio=mlp_ratios[0], drop_path=dpr[cur + i], sr_ratio=sr_ratios[0]) for i in range(depths[0])])
        self.norm1  = nn.LayerNorm(embed_dims[0], eps=1e-6)

        cur += depths[0]
        self.block2 = nn.ModuleList([Block(dim=embed_dims[1], num_heads=2, mlp_ratio=mlp_ratios[1], drop_path=dpr[cur + i], sr_ratio=sr_ratios[1]) for i in range(depths[1])])
        self.norm2  = nn.LayerNorm(embed_dims[1], eps=1e-6)

        cur += depths[1]
        self.block3 = nn.ModuleList([Block(dim=embed_dims[2], num_heads=5, mlp_ratio=mlp_ratios[2], drop_path=dpr[cur + i], sr_ratio=sr_ratios[2]) for i in range(depths[2])])
        self.norm3  = nn.LayerNorm(embed_dims[2], eps=1e-6)

        cur += depths[2]
        self.block4 = nn.ModuleList([Block(dim=embed_dims[3], num_heads=8, mlp_ratio=mlp_ratios[3], drop_path=dpr[cur + i], sr_ratio=sr_ratios[3]) for i in range(depths[3])])
        self.norm4  = nn.LayerNorm(embed_dims[3], eps=1e-6)

        state_dict:dict = torch.load(self.snapshot, map_location='cpu')
        state_dict.pop("head.weight")
        state_dict.pop("head.bias")
        self.load_state_dict(state_dict, strict=True)
        del state_dict

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed1', 'pos_embed2', 'pos_embed3', 'pos_embed4', 'cls_token'}  # has pos_embed may be better

    def forward(self, x):
        B = x.shape[0]
        # stage 1
        out1, H, W = self.patch_embed1(x)
        for i, blk in enumerate(self.block1):
            out1 = blk(out1, H, W)
        out1 = self.norm1(out1).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 2
        out2, H, W = self.patch_embed2(out1)
        for i, blk in enumerate(self.block2):
            out2 = blk(out2, H, W)
        out2 = self.norm2(out2).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 3
        out3, H, W = self.patch_embed3(out2)
        for i, blk in enumerate(self.block3):
            out3 = blk(out3, H, W)
        out3 = self.norm3(out3).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()

        # stage 4
        out4, H, W = self.patch_embed4(out3)
        for i, blk in enumerate(self.block4):
            out4 = blk(out4, H, W)
        out4 = self.norm4(out4).reshape(B, H, W, -1).permute(0, 3, 1, 2).contiguous()
        return out1, out2, out3, out4


from detectron2.modeling import BACKBONE_REGISTRY
from einops import rearrange, reduce, repeat
from .utils import VideoMultiscale_Shape, ImageMultiscale_Shape
import os
import time


@BACKBONE_REGISTRY.register()
class PVT_V2(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        pt_path = os.getenv('PT_PATH')
        pvt_v2 = PVT(embed_dims=[64, 128, 320, 512], mlp_ratios=[8, 8, 4, 4], 
                     depths=[3, 4, 6, 3], snapshot=os.path.join(pt_path, 'pvt_v2/pvt_v2_b2.pth'))
        self.pvt_v2 = pvt_v2

        freeze = configs['freeze']

        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)

        self.multiscale_shapes = {}
        for name, spatial_stride, dim  in zip(['res2', 'res3', 'res4', 'res5'], 
                                              [4, 8, 16, 32],
                                              [64, 128, 320, 512]):
            self.multiscale_shapes[name] =  ImageMultiscale_Shape(spatial_stride=spatial_stride, dim=dim)
        self.max_stride = 32

    def forward(self, x):

        if not self.training:
            batch_feats = []
            for haosen in x:
                feats =  self.pvt_v2(haosen.unsqueeze(0))
                batch_feats.append(feats)
            batch_feats = list(zip(*batch_feats)) # 4
            batch_feats = [torch.cat(haosen, dim=0) for haosen in batch_feats] # list[bt c h w]
            ret = {}
            names = ['res2', 'res3', 'res4', 'res5']
            for name, feat in zip(names, batch_feats):
                ret[name] = feat
            return ret            
        else:
            layer_outputs = self.pvt_v2(x)
            ret = {}
            names = ['res2', 'res3', 'res4', 'res5']
            for name, feat in zip(names, layer_outputs):
                ret[name] = feat
            return ret


    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

@BACKBONE_REGISTRY.register()
class Video2D_PVT_V2(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        self.image_homo = PVT_V2(configs=configs)

        self.multiscale_shapes = {}
        for name, temporal_stride, spatial_stride, dim  in zip(['res2', 'res3', 'res4', 'res5'],  
                                                               [1, 1, 1, 1], 
                                                               [4, 8, 16, 32],
                                                               [64, 128, 320, 512]):
            self.multiscale_shapes[name] =  VideoMultiscale_Shape(temporal_stride=temporal_stride, 
                                                                  spatial_stride=spatial_stride, dim=dim)
        self.max_stride = [1, 32]

    
    def forward(self, x):
        batch_size, _, T = x.shape[:3]
        x = rearrange(x, 'b c t h w -> (b t) c h w').contiguous()
        layer_outputs = self.image_homo(x)

        layer_outputs = {key: rearrange(value.contiguous(), '(b t) c h w -> b c t h w',b=batch_size, t=T).contiguous() \
                         for key, value in layer_outputs.items()}
        return layer_outputs
    
    def num_parameters(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)