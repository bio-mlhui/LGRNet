from inspect import isfunction
import math
import torch
from torch.nn.init import xavier_uniform_, constant_
import torch.nn.functional as F
from torch import nn, einsum
from einops import rearrange, repeat
from .functions import MSDeformAttnFunction
import warnings

def exists(val):
    return val is not None


def uniq(arr):
    return{el: True for el in arr}.keys()


def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d


def max_neg_value(t):
    return -torch.finfo(t.dtype).max


def init_(tensor):
    dim = tensor.shape[-1]
    std = 1 / math.sqrt(dim)
    tensor.uniform_(-std, std)
    return tensor


# feedforward
class GEGLU(nn.Module):
    def __init__(self, dim_in, dim_out):
        super().__init__()
        self.proj = nn.Linear(dim_in, dim_out * 2)

    def forward(self, x):
        x, gate = self.proj(x).chunk(2, dim=-1)
        return x * F.gelu(gate)


class FeedForward(nn.Module):
    def __init__(self, dim, dim_out=None, mult=4, glu=False, dropout=0.):
        super().__init__()
        inner_dim = int(dim * mult)
        dim_out = default(dim_out, dim)
        project_in = nn.Sequential(
            nn.Linear(dim, inner_dim),
            nn.GELU()
        ) if not glu else GEGLU(dim, inner_dim)

        self.net = nn.Sequential(
            project_in,
            nn.Dropout(dropout),
            nn.Linear(inner_dim, dim_out)
        )

    def forward(self, x):
        return self.net(x)


def zero_module(module):
    """
    Zero out the parameters of a module and return it.
    """
    for p in module.parameters():
        p.detach().zero_()
    return module


def Normalize(in_channels):
    return torch.nn.GroupNorm(num_groups=32, num_channels=in_channels, eps=1e-6, affine=True)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias = False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x)
        q, k, v = rearrange(qkv, 'b (qkv heads c) h w -> qkv b heads c (h w)', heads = self.heads, qkv=3)
        k = k.softmax(dim=-1)  
        context = torch.einsum('bhdn,bhen->bhde', k, v)
        out = torch.einsum('bhde,bhdn->bhen', context, q)
        out = rearrange(out, 'b heads c (h w) -> b (heads c) h w', heads=self.heads, h=h, w=w)
        return self.to_out(out)


class SpatialSelfAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.in_channels = in_channels

        self.norm = Normalize(in_channels)
        self.q = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.k = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.v = torch.nn.Conv2d(in_channels,
                                 in_channels,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)
        self.proj_out = torch.nn.Conv2d(in_channels,
                                        in_channels,
                                        kernel_size=1,
                                        stride=1,
                                        padding=0)

    def forward(self, x):
        h_ = x
        h_ = self.norm(h_)
        q = self.q(h_)
        k = self.k(h_)
        v = self.v(h_)

        # compute attention
        b,c,h,w = q.shape
        q = rearrange(q, 'b c h w -> b (h w) c')
        k = rearrange(k, 'b c h w -> b c (h w)')
        w_ = torch.einsum('bij,bjk->bik', q, k)

        w_ = w_ * (int(c)**(-0.5))
        w_ = torch.nn.functional.softmax(w_, dim=2)

        # attend to values
        v = rearrange(v, 'b c h w -> b c (h w)')
        w_ = rearrange(w_, 'b i j -> b j i')
        h_ = torch.einsum('bij,bjk->bik', v, w_)
        h_ = rearrange(h_, 'b c (h w) -> b c h w', h=h)
        h_ = self.proj_out(h_)

        return x+h_


class CrossAttention(nn.Module):
    def __init__(self, query_dim, context_dim=None, heads=8, dim_head=64, dropout=0.):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)

        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias=False)
        self.to_k = nn.Linear(context_dim, inner_dim, bias=False)
        self.to_v = nn.Linear(context_dim, inner_dim, bias=False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, query_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h=h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h=h)
            sim.masked_fill_(~mask, max_neg_value)

        # attention, what we cannot get enough of
        attn = sim.softmax(dim=-1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h=h)
        return self.to_out(out)

class BasicTransformerBlock(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True):
        super().__init__()
        self.attn1 = CrossAttention(query_dim=dim, heads=n_heads, dim_head=d_head, dropout=dropout)  # is a self-attention
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        self.attn2 = CrossAttention(query_dim=dim, context_dim=context_dim,
                                    heads=n_heads, dim_head=d_head, dropout=dropout)  # is self-attn if context is none
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.norm3 = nn.LayerNorm(dim)

    def forward(self, x, context=None):
        x = self.attn1(self.norm1(x)) + x
        x = self.attn2(self.norm2(x), context=context) + x
        x = self.ff(self.norm3(x)) + x
        return x


def _is_power_of_2(n):
    if (not isinstance(n, int)) or (n < 0):
        raise ValueError("invalid input for _is_power_of_2: {} (type: {})".format(n, type(n)))
    return (n & (n-1) == 0) and n != 0

class DeformAttn(nn.Module):
    def __init__(self, 
                 d_model=256,
                 nheads=8,
                 npoints=4,
                 nlevels=4,
                 key_dim=None):
        super().__init__()
        query_dim = d_model
        key_dim = d_model
        head_dim = d_model // nheads
        
        if d_model % nheads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, nheads))
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(head_dim):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")
        self.im2col_step = 64
        
        self.d_model = nheads * head_dim
        key_dim = default(key_dim, query_dim)
        
        self.value_proj = nn.Linear(key_dim, nheads * head_dim)
        self.sampling_offsets = nn.Linear(query_dim, nheads * nlevels * npoints * 2)
        
        self.attention_weights = nn.Linear(query_dim, nheads * nlevels * npoints)
        self.output_proj = nn.Linear(nheads * head_dim, query_dim)
        
        self.n_heads = nheads
        self.n_levels = nlevels
        self.head_dim = head_dim
        self.n_points = npoints
        self._reset_parameters()
    
    def _reset_parameters(self):
        constant_(self.sampling_offsets.weight.data, 0.)
        thetas = torch.arange(self.n_heads, dtype=torch.float32) * (2.0 * math.pi / self.n_heads)
        grid_init = torch.stack([thetas.cos(), thetas.sin()], -1)
        grid_init = (grid_init / grid_init.abs().max(-1, keepdim=True)[0]).view(self.n_heads, 1, 1, 2).repeat(1, self.n_levels, self.n_points, 1)
        for i in range(self.n_points):
            grid_init[:, :, i, :] *= i + 1
        with torch.no_grad():
            self.sampling_offsets.bias = nn.Parameter(grid_init.view(-1))
        constant_(self.attention_weights.weight.data, 0.)
        constant_(self.attention_weights.bias.data, 0.)
        xavier_uniform_(self.value_proj.weight.data)
        constant_(self.value_proj.bias.data, 0.)
        xavier_uniform_(self.output_proj.weight.data)
        constant_(self.output_proj.bias.data, 0.)        

    def forward(self, query, reference_points, 
                input_flatten, input_spatial_shapes, input_level_start_index,input_padding_mask=None):
        """
        multi-scale deformable attention, self attention if query == input_flatten
        Input:  
            - query: 
                T(b n c)
            - reference_points: center or reference boxes, normalized, [0, 1],  including padding area,
                                         add additional (w, h) to form reference boxes
                T(b n level 2) or T(b n level 4)
            - input_flatten: multi-scale特征
                T(b (h_\sigma w_\sigma) c)
            - input_spatial_shapes: 每个level的大小
                T(level 2)
            - input_level_start_index: [0, level1_start, level2_start]
            - input_padding_mask: True/False
                T(b), (h_\sigma w_\sigma))
        Output: 
            - query results:
                T(b, n c)
            - sampling_locations: normalized
                T(b, n, m*l*k, 2)
            - attention_weights: after softmax
                T(b, n, m*l*k)
        """
        batch_size, Nq, _ = query.shape
        _, Nk, _ = input_flatten.shape
        assert Nk == (input_spatial_shapes[:, 0] * input_spatial_shapes[:, 1]).sum()
        
        # B (h w) M * V
        value = self.value_proj(input_flatten)
        if input_padding_mask is not None:
            value = value.masked_fill(input_padding_mask[...,None], float(0))
        value = value.view(batch_size, Nk, self.n_heads, self.head_dim)
        sampling_offesets = self.sampling_offsets(query).view(batch_size, Nq, self.n_heads,  self.n_levels, self.n_points, 2)
        attention_weights= self.attention_weights(query).view(batch_size, Nq, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, dim=-1).view(batch_size, Nq, self.n_heads, self.n_levels, self.n_points)
        # b, n ,head, level, point, 2
        if reference_points.shape[-1] == 2:
            # T(2 level)
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] + \
                sampling_offesets / offset_normalizer[None, None, None, :, None,:]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] + \
                sampling_offesets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise NotImplementedError
        output = MSDeformAttnFunction.apply(value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)
        
        return output, sampling_locations, attention_weights
            
        
class ContextuallSelfAttention(nn.Module):
    def __init__(self, 
                 d_model, 
                 n_points,
                 n_heads,
                 context_dim=None):
        super().__init__()
        context_dim = default(context_dim, d_model)
        query_dim = key_dim = d_model
        
        if d_model % n_heads != 0:
            raise ValueError('d_model must be divisible by n_heads, but got {} and {}'.format(d_model, n_heads))
        head_dim = d_model // n_heads
        # you'd better set _d_per_head to a power of 2 which is more efficient in our CUDA implementation
        if not _is_power_of_2(head_dim):
            warnings.warn("You'd better set d_model in MSDeformAttn to make the dimension of each attention head a power of 2 "
                          "which is more efficient in our CUDA implementation.")
        self.im2col_step = 64
        
        self.d_model = d_model
        self.nheads = n_heads
        self.head_dim = head_dim
        self.npoints = n_points
        
        self.nlevels = 1
        
        self.value_proj = nn.Linear(key_dim, n_heads * head_dim)
        self.sampling_offsets = nn.Linear(query_dim, n_heads * n_points * 2)
        self.attention_weights = nn.Linear(query_dim, n_heads * n_points)
        
        self.output_proj = nn.Linear(n_heads * head_dim, query_dim)
        
    def forward(self,
                context, context_mask,
                query, reference_points, query_padding_mask = None,):
        """
        contextual deformable attention
        Input:  
            - context:
                T(b n c)
            - context_mask:
                T(b n)
            - query: 
                T(b (h w) c)
            - reference_points: center or reference boxes, normalized, [0, 1],  including padding area,
                T(b (h w) 2/4)
            - query_padding_mask:
                T(b (h w))
        Output: 
            - query results:
                T(b, n c)
            - sampling_locations: normalized
                T(b, n, m*l*k, 2)
            - attention_weights: after softmax
                T(b, n, m*l*k)
        """
        key = query
        key_padding_mask = query_padding_mask
        batch_size, Nq, _ = query.shape
        Nk = Nq
        
        input_spatial_shapes = torch.tensor(query.shape[-2:]).unsqueeze(0)  # T(1, 2)
        input_level_start_index = [0, ]
        
        # B (h w) M * V
        value = self.value_proj(key)
        if key_padding_mask is not None:
            value = value.masked_fill(key_padding_mask[..., None], float(0))
        value = value.view(batch_size, Nk, self.nheads, self.head_dim)
        
        sampling_offesets = self.sampling_offsets(query).view(batch_size, Nq, self.n_heads,  self.n_levels, self.n_points, 2)
        attention_weights= self.attention_weights(query).view(batch_size, Nq, self.n_heads, self.n_levels * self.n_points)
        attention_weights = F.softmax(attention_weights, dim=-1).view(batch_size, Nq, self.n_heads, self.n_levels, self.n_points)
        # b, n ,head, level, point, 2
        if reference_points.shape[-1] == 2:
            # T(2 level)
            offset_normalizer = torch.stack([input_spatial_shapes[..., 1], input_spatial_shapes[..., 0]], -1)
            sampling_locations = reference_points[:, :, None, :, None, :] + \
                sampling_offesets / offset_normalizer[None, None, None, :, None,:]
        elif reference_points.shape[-1] == 4:
            sampling_locations = reference_points[:, :, None, :, None, :2] + \
                sampling_offesets / self.n_points * reference_points[:, :, None, :, None, 2:] * 0.5
        else:
            raise NotImplementedError
        output = MSDeformAttnFunction.apply(value, input_spatial_shapes, input_level_start_index, sampling_locations, attention_weights, self.im2col_step)
        output = self.output_proj(output)
        
        return output, sampling_locations, attention_weights

class BasicTransformerBlock_v2(nn.Module):
    def __init__(self, dim, n_heads, d_head, dropout=0., context_dim=None, gated_ff=True):
        super().__init__()
        self.ff = FeedForward(dim, dropout=dropout, glu=gated_ff)
        
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)

    def forward(self, x, context=None):
        x = self.norm1(self.attn1(x, context=context) + x)
        x = self.norm2(self.ff(x) + x)
        return x



class SpatialTransformer(nn.Module):
    """
    Transformer block for image-like data.
    First, project the input (aka embedding)
    and reshape to b, t, d.
    Then apply standard transformer action.
    Finally, reshape to image
    """
    def __init__(self, in_channels, n_heads, d_head,
                 depth=1, dropout=0., context_dim=None):
        super().__init__()
        self.in_channels = in_channels
        inner_dim = n_heads * d_head
        self.norm = Normalize(in_channels)

        self.proj_in = nn.Conv2d(in_channels,
                                 inner_dim,
                                 kernel_size=1,
                                 stride=1,
                                 padding=0)

        self.transformer_blocks = nn.ModuleList(
            [BasicTransformerBlock(inner_dim, n_heads, d_head, dropout=dropout, context_dim=context_dim)
                for d in range(depth)]
        )

        self.proj_out = zero_module(nn.Conv2d(inner_dim,
                                              in_channels,
                                              kernel_size=1,
                                              stride=1,
                                              padding=0))

    def forward(self, x, context=None):
        # note: if no context is given, cross-attention defaults to self-attention
        b, c, h, w = x.shape
        x_in = x
        x = self.norm(x)
        x = self.proj_in(x)
        x = rearrange(x, 'b c h w -> b (h w) c')
        for block in self.transformer_blocks:
            x = block(x, context=context)
        x = rearrange(x, 'b (h w) c -> b c h w', h=h, w=w)
        x = self.proj_out(x)
        return x + x_in
    