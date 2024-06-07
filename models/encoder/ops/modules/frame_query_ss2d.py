
from models.layers.position_encoding import build_position_encoding
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
import math
import warnings
import math

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.init import xavier_uniform_, constant_

from ..functions import MSDeformAttnFunction

from mamba_ssm import Mamba
from einops import rearrange, reduce, repeat
from detectron2.modeling import META_ARCH_REGISTRY

# v1
class SS2D(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        # d_state="auto", # 20240109
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=4, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=4, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True) # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True) # (K=4, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn
        
        B, C, H, W = x.shape
        L = H * W
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)], dim=1).view(B, 2, -1, L)
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1) # (b, k, d, l)

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L)
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L)

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y

    def forward(self, x: torch.Tensor, **kwargs):
        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)
        y1, y2, y3, y4 = self.forward_core(x) # B C hw
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1)
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out

class SS2D_FrameQuery(nn.Module):
    def __init__(self, configs,):
        super().__init__()
        d_model = configs['d_model']
        self.homo = SS2D(d_model=configs['d_model'],
                        d_state=configs['d_state'] if 'd_state' in configs else 16,
                        d_conv=configs['d_conv'] if 'd_conv' in configs else 3,
                        expand=configs['expand'] if 'expand' in configs else 2,
                        dt_rank=configs['dt_rank'] if 'dt_rank' in configs else 'auto',
                        dt_min=configs['dt_min'] if 'dt_min' in configs else 0.001,
                        dt_max=configs['dt_max'] if 'dt_max' in configs else 0.1,
                        dt_init=configs['dt_init'] if 'dt_init' in configs else 'random',
                        dt_scale=configs['dt_scale'] if 'dt_scale' in configs else 1.0,
                        dt_init_floor=configs['dt_init_floor'] if 'dt_init_floor' in configs else 1e-4,
                        dropout=configs['dropout'] if 'dropout' in configs else 0,
                        conv_bias=configs['conv_bias'] if 'conv_bias' in configs else True,
                        bias=configs['bias'] if 'bias' in configs else False,
                        )
        self.pos_1d = build_position_encoding(position_embedding_name='1d')  # t上的position embedding

    def forward(self, 
                frame_query_feats=None,  # n bt c  
                frame_query_poses=None, # n bt c  # nq上的Position embedding
                nf=None,
                **kwargs
                ):
        batch_size = frame_query_feats.shape[1] // nf # b
        frame_query_feats += frame_query_poses  
        frame_query_feats = rearrange(frame_query_feats, 'n (b t) c -> b t n c',b=batch_size,t=nf).contiguous()

        sin_poses = self.pos_1d(torch.zeros_like(frame_query_feats[..., 0].permute(0, 2, 1).flatten(0, 1)).bool(), 
                                hidden_dim=frame_query_feats.shape[-1]) # bn c t
        sin_poses = rearrange(sin_poses, '(b n) c t -> b t n c', b=batch_size)
        frame_query_feats += sin_poses

        frame_query_feats = self.homo(frame_query_feats) # b t n c
        
        frame_query_feats = frame_query_feats.permute(2, 0, 1, 3).flatten(1, 2).contiguous() # n bt c

        return frame_query_feats, None

@META_ARCH_REGISTRY.register()
class FrameQuery_SS2DLayer(nn.Module):
    def __init__(self, 
                 configs, 
                 dropout=0.0,
                 activation="relu", normalize_before=False):
        super().__init__()
        d_model = configs['d_model']
        dropout = configs['dropout']
        self.self_attn = SS2D_FrameQuery(configs)
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

        from models.layers.decoder_layers import FFNLayer
        self.ffn = FFNLayer(d_model=d_model,
                            dim_feedforward=configs['dim_feedforward'],
                            dropout=dropout,)

    def forward(self, 
                frame_query_feats,  # n bt c  
                frame_query_poses, # n bt c  # nq上的Position embedding
                nf=None,
                **kwargs):
        
        tgt2 = self.self_attn(frame_query_feats=frame_query_feats,  # n bt c  
                             frame_query_poses=frame_query_poses, # n bt c  # nq上的Position embedding
                             nf=nf,)[0]
        frame_query_feats = frame_query_feats + self.dropout(tgt2)
        frame_query_feats = self.norm(frame_query_feats)

        frame_query_feats = self.ffn(frame_query_feats)
        return frame_query_feats

from models.layers.decoder_layers import CrossAttentionLayer, SelfAttentionLayer, FFNLayer, FFNLayer_mlpRatio
@META_ARCH_REGISTRY.register()
class TemporalQuery_CrossSelf(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        d_model = configs['d_model']
        attn_configs = configs['attn']
        self.cross_layers = CrossAttentionLayer(d_model=d_model,
                                                nhead=attn_configs['nheads'],
                                                dropout=0.0,
                                                normalize_before=attn_configs['normalize_before'])
        self.self_layers = SelfAttentionLayer(d_model=d_model,
                                                nhead=attn_configs['nheads'],
                                                dropout=0.0,
                                                normalize_before=attn_configs['normalize_before'])
        self.ffn_layers = FFNLayer(d_model=d_model,
                                    dim_feedforward=attn_configs['dim_feedforward'],
                                    dropout=0.0,
                                    normalize_before=attn_configs['normalize_before'])
    
    def forward(self, 
                temporal_query_feats, 
                temporal_query_poses,
                frame_query_feats, frame_query_poses,
                video_aux_dict=None, **kwargs):
        # nq b c; nq bt c
        nq, batch_size, _ = temporal_query_feats.shape
        nf = frame_query_feats.shape[1] // batch_size
        nqf = frame_query_feats.shape[0]
        frame_query_feats = rearrange(frame_query_feats, 'nq (b t) c -> (t nq) b c',b=batch_size, t=nf)
        frame_query_poses = rearrange(frame_query_poses, 'nq (b t) c -> (t nq) b c',b=batch_size, t=nf)
        temporal_query_feats = self.cross_layers(
            tgt=temporal_query_feats, # n b c
            memory=frame_query_feats,  # t nqf b c
            pos=frame_query_poses, 
            query_pos=temporal_query_poses,
        )
        temporal_query_feats = self.self_layers(
            temporal_query_feats,
            query_pos=temporal_query_poses,
        )
        temporal_query_feats = self.ffn_layers(
            temporal_query_feats 
        )
        return temporal_query_feats
    
# v2 多层
class SS2D_FrameQuery_v2(nn.Module):
    def __init__(self, configs,):
        super().__init__()
        d_model = configs['d_model']
        self.homo = SS2D(d_model=configs['d_model'],
                        d_state=configs['d_state'] if 'd_state' in configs else 16,
                        d_conv=configs['d_conv'] if 'd_conv' in configs else 3,
                        expand=configs['expand'] if 'expand' in configs else 2,
                        dt_rank=configs['dt_rank'] if 'dt_rank' in configs else 'auto',
                        dt_min=configs['dt_min'] if 'dt_min' in configs else 0.001,
                        dt_max=configs['dt_max'] if 'dt_max' in configs else 0.1,
                        dt_init=configs['dt_init'] if 'dt_init' in configs else 'random',
                        dt_scale=configs['dt_scale'] if 'dt_scale' in configs else 1.0,
                        dt_init_floor=configs['dt_init_floor'] if 'dt_init_floor' in configs else 1e-4,
                        dropout=configs['dropout'] if 'dropout' in configs else 0,
                        conv_bias=configs['conv_bias'] if 'conv_bias' in configs else True,
                        bias=configs['bias'] if 'bias' in configs else False,
                        )
        self.pos_1d = build_position_encoding(position_embedding_name='1d')  # t上的position embedding
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(configs['dropout'])

    def forward(self, 
                frame_query_feats=None,  # n bt c  
                frame_query_poses=None, # n bt c  # nq上的Position embedding
                nf=None,
                **kwargs
                ):
        batch_size = frame_query_feats.shape[1] // nf # b
        tgt2 = frame_query_feats + frame_query_poses  
        tgt2 = rearrange(tgt2, 'n (b t) c -> b t n c',b=batch_size,t=nf).contiguous()

        sin_poses = self.pos_1d(torch.zeros_like(tgt2[..., 0].permute(0, 2, 1).flatten(0, 1)).bool(), 
                                hidden_dim=tgt2.shape[-1]) # bn c t
        sin_poses = rearrange(sin_poses, '(b n) c t -> b t n c', b=batch_size)
        tgt2 += sin_poses

        tgt2 = self.homo(tgt2) # b t n c
        
        tgt2 = tgt2.permute(2, 0, 1, 3).flatten(1, 2).contiguous() # n bt c


        frame_query_feats = frame_query_feats + self.dropout(tgt2)
        frame_query_feats = self.norm(frame_query_feats)

        return frame_query_feats, None
    
from models.layers.utils import _get_clones
@META_ARCH_REGISTRY.register()
class FrameQuery_SS2DLayer_v2(nn.Module):
    def __init__(self, 
                 configs, 
                 dropout=0.0):
        super().__init__()
        d_model = configs['d_model']
        n_layers = configs['nlayers'] if 'nlayers' in configs else 1
        self.nlayers = n_layers
        self.self_attn = _get_clones(SS2D_FrameQuery_v2(configs), n_layers)

        from models.layers.decoder_layers import FFNLayer
        self.ffn = FFNLayer(d_model=d_model,
                            dim_feedforward=configs['dim_feedforward'],
                            dropout=configs['dropout'],)

    def forward(self, 
                frame_query_feats,  # n bt c  
                frame_query_poses, # n bt c  # nq上的Position embedding
                nf=None,
                **kwargs):
        
        for i in range(self.nlayers):
            frame_query_feats = self.self_attn[i](frame_query_feats=frame_query_feats,  # n bt c  
                                                  frame_query_poses=frame_query_poses, # n bt c  # nq上的Position embedding
                                                  nf=nf,)[0]
            
        frame_query_feats = self.ffn(frame_query_feats)
        return frame_query_feats

@META_ARCH_REGISTRY.register()
class TemporalQuery_CrossSelf_v2(nn.Module):
    def __init__(self, configs) -> None:
        super().__init__()
        d_model = configs['d_model']
        attn_configs = configs['attn']
        self.cross_layers = CrossAttentionLayer(d_model=d_model,
                                                nhead=attn_configs['nheads'],
                                                dropout=0.0,
                                                normalize_before=attn_configs['normalize_before'])
        self.self_layers = SelfAttentionLayer(d_model=d_model,
                                                nhead=attn_configs['nheads'],
                                                dropout=0.0,
                                                normalize_before=attn_configs['normalize_before'])
        self.ffn_layers = FFNLayer_mlpRatio(d_model=d_model,
                                            mlp_ratio=attn_configs['ffn_mlp_ratio'],
                                            dropout=0.0,
                                            normalize_before=attn_configs['normalize_before'])
    
    def forward(self, 
                temporal_query_feats, 
                temporal_query_poses,
                frame_query_feats, frame_query_poses,
                video_aux_dict=None, **kwargs):
        # nq b c; nq bt c
        nq, batch_size, _ = temporal_query_feats.shape
        nf = frame_query_feats.shape[1] // batch_size
        nqf = frame_query_feats.shape[0]
        frame_query_feats = rearrange(frame_query_feats, 'nq (b t) c -> (t nq) b c',b=batch_size, t=nf)
        frame_query_poses = rearrange(frame_query_poses, 'nq (b t) c -> (t nq) b c',b=batch_size, t=nf)
        temporal_query_feats = self.cross_layers(
            tgt=temporal_query_feats, # n b c
            memory=frame_query_feats,  # t nqf b c
            pos=frame_query_poses, 
            query_pos=temporal_query_poses,
        )
        temporal_query_feats = self.self_layers(
            temporal_query_feats,
            query_pos=temporal_query_poses,
        )
        temporal_query_feats = self.ffn_layers(
            temporal_query_feats 
        )
        return temporal_query_feats


class Hilbert_2DSelectiveScan(nn.Module):
    def __init__(
        self,
        d_model,
        d_state=16,
        # d_state="auto", # 20240109
        d_conv=3,
        expand=2,
        dt_rank="auto",
        dt_min=0.001,
        dt_max=0.1,
        dt_init="random",
        dt_scale=1.0,
        dt_init_floor=1e-4,
        dropout=0.,
        conv_bias=True,
        bias=False,
        device=None,
        dtype=None,
        scan_order=None,
        **kwargs,
    ):
        factory_kwargs = {"device": device, "dtype": dtype}
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        # self.d_state = math.ceil(self.d_model / 6) if d_state == "auto" else d_model # 20240109
        self.d_conv = d_conv
        self.expand = expand
        self.d_inner = int(self.expand * self.d_model)
        self.dt_rank = math.ceil(self.d_model / 16) if dt_rank == "auto" else dt_rank

        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias, **factory_kwargs)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,
            out_channels=self.d_inner,
            groups=self.d_inner,
            bias=conv_bias,
            kernel_size=d_conv,
            padding=(d_conv - 1) // 2,
            **factory_kwargs,
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False, **factory_kwargs), 
        )
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0)) # (K=2, N, inner)
        del self.x_proj

        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor, **factory_kwargs),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0)) # (K=2, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0)) # (K=2, inner)
        del self.dt_projs
        
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=2, merge=True) # (K=2, D, N)
        self.Ds = self.D_init(self.d_inner, copies=2, merge=True) # (K=2, D, N)

        # self.selective_scan = selective_scan_fn
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias, **factory_kwargs)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None
        self.scan_order = scan_order

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4, **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank**-0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True
        
        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor, hilbert_curve):
        # LongTensor[int] 按照hw进行flatten之后的hilbert排序
        self.selective_scan = selective_scan_fn
                
        B, C, H, W = x.shape
        L = H * W
        K = 2

        if self.scan_order == 'zigzag':
            x_hw = x.view(B, -1, L).contiguous() # b c hw
            xs = torch.stack([x_hw, torch.flip(x_hw, dims=[-1])], dim=1) # (b, k, d, l)
        elif self.scan_order == 'hilbert':
            x_hw = x.flatten(2).contiguous() # b c hw
            x_hil = x_hw.index_select(dim=-1, index=hilbert_curve)
            xs = torch.stack([x_hil, torch.flip(x_hil, dims=[-1])], dim=1) # (b, k, d, l)
        else:
            raise ValueError()

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
        # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
        # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

        xs = xs.float().view(B, -1, L) # (b, k * d, l)
        dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
        Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1) # (k * d)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
        dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

        out_y = self.selective_scan(
            xs, dts, 
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)

        assert out_y.dtype == torch.float

        if self.scan_order == 'zigzag':
            hw_order = out_y[:, 0].contiguous().view(B, -1, H, W).contiguous()
            rhw_order = torch.flip(out_y[:, 1].contiguous(), dims=[-1]).contiguous()
            rhw_order = rhw_order.view(B, -1, H, W,).contiguous()
            return hw_order + rhw_order
        
        elif self.scan_order == 'hilbert':
            hil_order = out_y[:, 0].contiguous() # b c hw
            rhil_order = torch.flip(out_y[:, 1].contiguous(), dims=[-1]).contiguous() # b c hw

            sum_out = torch.zeros_like(hil_order)
            hilbert_curve = repeat(hilbert_curve, 'hw -> b c hw', b=hil_order.shape[0], c=hil_order.shape[1])
            assert hil_order.shape == hilbert_curve.shape
            sum_out.scatter_add_(dim=-1, index=hilbert_curve, src=hil_order)
            sum_out.scatter_add_(dim=-1, index=hilbert_curve, src=rhil_order)
            sum_out = sum_out.view(B, -1, H, W).contiguous()
            return sum_out


    # def forward_corev0(self, x: torch.Tensor, hilbert_curve):
    #     # LongTensor[int] 按照hw进行flatten之后的hilbert排序
    #     self.selective_scan = selective_scan_fn
                
    #     B, C, H, W, T = x.shape
    #     L = H * W * T
    #     K = 2

    #     if self.scan_order == 'zigzag':
    #         x_hw = x.view(B, -1, L).contiguous() # b c hwt
    #         xs = torch.stack([x_hw, torch.flip(x_hw, dims=[-1])], dim=1) # (b, k, d, l)
    #     elif self.scan_order == 'hilbert':
    #         x_hw = x.flatten(2).contiguous() # b c hwt
    #         x_hil = x_hw.index_select(dim=-1, index=hilbert_curve)
    #         xs = torch.stack([x_hil, torch.flip(x_hil, dims=[-1])], dim=1) # (b, k, d, l)
    #     else:
    #         raise ValueError()

    #     x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight)
    #     # x_dbl = x_dbl + self.x_proj_bias.view(1, K, -1, 1)
    #     dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2)
    #     dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight)
    #     # dts = dts + self.dt_projs_bias.view(1, K, -1, 1)

    #     xs = xs.float().view(B, -1, L) # (b, k * d, l)
    #     dts = dts.contiguous().float().view(B, -1, L) # (b, k * d, l)
    #     Bs = Bs.float().view(B, K, -1, L) # (b, k, d_state, l)
    #     Cs = Cs.float().view(B, K, -1, L) # (b, k, d_state, l)
    #     Ds = self.Ds.float().view(-1) # (k * d)
    #     As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  # (k * d, d_state)
    #     dt_projs_bias = self.dt_projs_bias.float().view(-1) # (k * d)

    #     out_y = self.selective_scan(
    #         xs, dts, 
    #         As, Bs, Cs, Ds, z=None,
    #         delta_bias=dt_projs_bias,
    #         delta_softplus=True,
    #         return_last_state=False,
    #     ).view(B, K, -1, L)

    #     assert out_y.dtype == torch.float

    #     if self.scan_order == 'zigzag':
    #         hw_order = out_y[:, 0].contiguous().view(B, -1, H, W).contiguous()
    #         rhw_order = torch.flip(out_y[:, 1].contiguous(), dims=[-1]).contiguous()
    #         rhw_order = rhw_order.view(B, -1, H, W,).contiguous()
    #         return hw_order + rhw_order
        
    #     elif self.scan_order == 'hilbert':
    #         hil_order = out_y[:, 0].contiguous() # b c hw
    #         rhil_order = torch.flip(out_y[:, 1].contiguous(), dims=[-1]).contiguous() # b c hw

    #         sum_out = torch.zeros_like(hil_order)
    #         hilbert_curve = repeat(hilbert_curve, 'hwt -> b c hwt', b=hil_order.shape[0], c=hil_order.shape[1])
    #         assert hil_order.shape == hilbert_curve.shape
    #         sum_out.scatter_add_(dim=-1, index=hilbert_curve, src=hil_order)
    #         sum_out.scatter_add_(dim=-1, index=hilbert_curve, src=rhil_order)
    #         sum_out = sum_out.view(B, -1, H, W).contiguous()
    #         return sum_out


    def forward(self, x: torch.Tensor, hilbert_curve, **kwargs):

        B, H, W, C = x.shape
        xz = self.in_proj(x)
        x, z = xz.chunk(2, dim=-1) # (b, h, w, d)

        x = x.permute(0, 3, 1, 2).contiguous()
        x = self.act(self.conv2d(x)) # (b, d, h, w)
        
        y  = self.forward_core(x, hilbert_curve=hilbert_curve) # B C h w
        y = y.permute(0, 2, 3, 1).contiguous() # b h w c
        y = self.out_norm(y)
        y = y * F.silu(z)
        out = self.out_proj(y)
        if self.dropout is not None:
            out = self.dropout(out)
        return out


class SS2D_FrameQuery_hilbert(nn.Module):
    def __init__(self, configs,):
        super().__init__()
        d_model = configs['d_model']
        self.homo = Hilbert_2DSelectiveScan(d_model=configs['d_model'],
                        d_state=configs['d_state'] if 'd_state' in configs else 16,
                        d_conv=configs['d_conv'] if 'd_conv' in configs else 3,
                        expand=configs['expand'] if 'expand' in configs else 2,
                        dt_rank=configs['dt_rank'] if 'dt_rank' in configs else 'auto',
                        dt_min=configs['dt_min'] if 'dt_min' in configs else 0.001,
                        dt_max=configs['dt_max'] if 'dt_max' in configs else 0.1,
                        dt_init=configs['dt_init'] if 'dt_init' in configs else 'random',
                        dt_scale=configs['dt_scale'] if 'dt_scale' in configs else 1.0,
                        dt_init_floor=configs['dt_init_floor'] if 'dt_init_floor' in configs else 1e-4,
                        dropout=configs['dropout'] if 'dropout' in configs else 0,
                        conv_bias=configs['conv_bias'] if 'conv_bias' in configs else True,
                        bias=configs['bias'] if 'bias' in configs else False,
                        scan_order=configs['scan_order']
                        )
        self.pos_1d = build_position_encoding(position_embedding_name='1d')  # t上的position embedding
        self.norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(configs['dropout'])

    def forward(self, 
                frame_query_feats=None,  # n bt c  
                frame_query_poses=None, # n bt c  # nq上的Position embedding
                hilbert_curve=None,
                nf=None,
                **kwargs
                ):
        batch_size = frame_query_feats.shape[1] // nf # b
        tgt2 = frame_query_feats + frame_query_poses  
        tgt2 = rearrange(tgt2, 'n (b t) c -> b t n c',b=batch_size,t=nf).contiguous()

        sin_poses = self.pos_1d(torch.zeros_like(tgt2[..., 0].permute(0, 2, 1).flatten(0, 1)).bool(), 
                                hidden_dim=tgt2.shape[-1]) # bn c t
        sin_poses = rearrange(sin_poses, '(b n) c t -> b t n c', b=batch_size)
        tgt2 += sin_poses

        tgt2 = self.homo(tgt2, hilbert_curve=hilbert_curve) # b t n c
        
        tgt2 = tgt2.permute(2, 0, 1, 3).flatten(1, 2).contiguous() # n bt c


        frame_query_feats = frame_query_feats + self.dropout(tgt2)
        frame_query_feats = self.norm(frame_query_feats)

        return frame_query_feats, None
    
from models.layers.utils import _get_clones

@META_ARCH_REGISTRY.register()
class FrameQuery_SS2DLayer_hilbert(nn.Module):
    def __init__(self, 
                 configs, 
                 dropout=0.0):
        super().__init__()
        d_model = configs['d_model']
        n_layers = configs['nlayers'] if 'nlayers' in configs else 1
        self.nlayers = n_layers
        self.self_attn = _get_clones(SS2D_FrameQuery_hilbert(configs), n_layers)
        from models.layers.decoder_layers import FFNLayer
        self.ffn = FFNLayer(d_model=d_model,
                            dim_feedforward=configs['dim_feedforward'],
                            dropout=configs['dropout'],)

    def forward(self, 
                frame_query_feats,  # n bt c  
                frame_query_poses, # n bt c  # nq上的Position embedding
                video_aux_dict=None,
                **kwargs):

        for i in range(self.nlayers):
            frame_query_feats = self.self_attn[i](frame_query_feats=frame_query_feats,  # n bt c  
                                                  frame_query_poses=frame_query_poses, # n bt c  # nq上的Position embedding
                                                  nf=video_aux_dict['nf'],
                                                  hilbert_curve=video_aux_dict['hilbert_curve'])[0]
        frame_query_feats = self.ffn(frame_query_feats)
        return frame_query_feats



