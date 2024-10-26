import torch
import torch.nn as nn
import torch.nn.functional as F

from einops import rearrange
from functools import partial

import matplotlib.pyplot as plt
import math
from inspect import isfunction


def exists(val):
    return val is not None

def default(val, d):
    if exists(val):
        return val
    return d() if isfunction(d) else d

class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x

def Upsample(dim):
    return nn.ConvTranspose2d(dim, dim, 4, 2, 1)

def Downsample(dim):
    return nn.Conv2d(dim, dim, 4, 2, 1)


class SinusoidalPositionEmbeddings(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, time):
        device = time.device
        half_dim = self.dim // 2
        embeddings = math.log(10000) / (half_dim - 1)
        embeddings = torch.exp(torch.arange(half_dim, device=device) * -embeddings)
        embeddings = time[:, None] * embeddings[None, :]
        embeddings = torch.cat((embeddings.sin(), embeddings.cos()), dim=-1)
        return embeddings
    
    
class ConvNextBlock(nn.Module):

    def __init__(self, dim, dim_out, *, time_emb_dim=None, mult=2, norm=True):
        super().__init__()
        self.mlp = (
            nn.Sequential(nn.GELU(), nn.Linear(time_emb_dim, dim))
            if time_emb_dim is not None else None
        )

        self.ds_conv = nn.Conv2d(dim, dim, 7, padding=3, groups=dim)

        self.net = nn.Sequential(
            nn.GroupNorm(1, dim) if norm else nn.Identity(),
            nn.Conv2d(dim, dim_out * mult, 3, padding=1),
            nn.GELU(),
            nn.GroupNorm(1, dim_out * mult),
            nn.Conv2d(dim_out * mult, dim_out, 3, padding=1),
        )

        self.res_conv = nn.Conv2d(dim, dim_out, 1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None):
        h = self.ds_conv(x)

        if self.mlp is not None and time_emb is not None:
            assert time_emb is not None, "time embedding must be passed in"
            condition = self.mlp(time_emb)
            h = h + rearrange(condition, "b c -> b c 1 1")

        h = self.net(h)
        return h + self.res_conv(x)

class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_q = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.to_k = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.to_v = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x, context=None): #  x:(b,h*w,c), context:(b,seq_len,context_dim)
        b, c, h, w = x.shape
        q = self.to_q(x) # q:(b,h*w,inner_dim)
        context = default(context, x)

        k = self.to_k(context)
        v = self.to_v(context)

        # qkv = self.to_qkv(x).chunk(3, dim=1)
        # q, k, v = map(
        #     lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        # )

        # Rearrange q, k, v for multi-head attention
        q = rearrange(q, 'b (h c) x y -> b h c (x y)', h=self.heads) * self.scale
        k = rearrange(k, 'b (h c) x y -> b h c (x y)', h=self.heads)
        v = rearrange(v, 'b (h c) x y -> b h c (x y)', h=self.heads)

        sim = torch.einsum("b h d i, b h d j -> b h i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("b h i j, b h d j -> b h i d", attn, v)
        out = rearrange(out, "b h (x y) d -> b (h d) x y", x=h, y=w)
        return self.to_out(out)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_q = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.to_k = nn.Conv2d(dim, hidden_dim, 1, bias=False)
        self.to_v = nn.Conv2d(dim, hidden_dim, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    nn.GroupNorm(1, dim))

    def forward(self, x, context=None):
        b, c, h, w = x.shape
        # print('inside LinearAttention class now')
        # print('x.shape:', x.shape)
        # if (context is not None):
            # print('context.shape:', context.shape)
        q = self.to_q(x) # q:(b,h*w,inner_dim)
        context = default(context, x)
        k = self.to_k(context)
        v = self.to_v(context)

        # q, k, v = map(
        #     lambda t: rearrange(t, "b (h c) x y -> b h c (x y)", h=self.heads), qkv
        # )

        q = rearrange(q, 'b (h c) x y -> b h c (x y)', h=self.heads)
        k = rearrange(k, 'b (h c) x y -> b h c (x y)', h=self.heads)
        v = rearrange(v, 'b (h c) x y -> b h c (x y)', h=self.heads)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        context_kv = torch.einsum("b h d n, b h e n -> b h d e", k, v)

        out = torch.einsum("b h d e, b h d n -> b h e n", context_kv, q)
        out = rearrange(out, "b h c (x y) -> b (h c) x y", h=self.heads, x=h, y=w)
        return self.to_out(out)
    



class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = nn.GroupNorm(1, dim)

    def forward(self, x, *args, **kwargs):
        x = self.norm(x)
        return self.fn(x, *args, **kwargs)




class Unet(nn.Module):
    def __init__(
        self,
        dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=3,
        with_time_emb=True,
        resnet_block_groups=8,
        use_convnext=True,
        convnext_mult=2,
    ):
        super().__init__()

        # determine dimensions
        self.channels = channels

        init_dim = init_dim if init_dim is not None else dim // 3 * 2
        self.init_conv = nn.Conv2d(channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: int(dim * m), dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        if use_convnext:
            block_klass = partial(ConvNextBlock, mult=convnext_mult)
        else:
            raise NotImplementedError()

        # time embeddings
        if with_time_emb:
            time_dim = dim * 4
            self.time_mlp = nn.Sequential(
                SinusoidalPositionEmbeddings(dim),
                nn.Linear(dim, time_dim),
                nn.GELU(),
                nn.Linear(time_dim, time_dim),
            )

        else:
            time_dim = None
            self.time_mlp = None
            self.cond_mlp = None
            self.sim_mlp = None

        # layers
        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList(
                    [
                        block_klass(dim_in, dim_out, time_emb_dim=time_dim),
                        block_klass(dim_out, dim_out, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                        Downsample(dim_out) if not is_last else nn.Identity(),
                    ]
                )
            )

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)
        self.mid_cross_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim, time_emb_dim=time_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out[1:])):
            is_last = ind >= (num_resolutions - 1)

            self.ups.append(
                nn.ModuleList(
                    [
                        block_klass(dim_out * 2, dim_in, time_emb_dim=time_dim),
                        block_klass(dim_in, dim_in, time_emb_dim=time_dim),
                        Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                        Upsample(dim_in) if not is_last else nn.Identity(),
                    ]
                )
            )

        out_dim = out_dim if out_dim is not None else channels
        self.final_conv = nn.Sequential(
            block_klass(dim, dim), nn.Conv2d(dim, out_dim, 1)
        )

    def forward(self, x, time, context):
        
        intermediate_outputs = {}
        x = self.init_conv(x)


        t = self.time_mlp(time) if self.time_mlp is not None else None

        h = []
        encoder_interm_outputs = []
        decoder_interm_outputs = []



        # downsample
        down_counter = 0
        for block1, block2, cross_attn, downsample in self.downs:
            suffix_key = str(down_counter)
            x = block1(x, t)
            intermediate_outputs['downsample_1' + suffix_key] = x
            x = block2(x, t)
            intermediate_outputs['downsample_2' + suffix_key] = x

            x = cross_attn(x, context=None)
            intermediate_outputs['down_cross_attn' + suffix_key] = x

            h.append(x)
            x = downsample(x)
            intermediate_outputs['downsample_fin' + suffix_key] = x

            down_counter += 1


        # bottleneck
        suffix_key = 'bridge'
        x = self.mid_block1(x, t)
        intermediate_outputs['mid1' + suffix_key] = x
        x = self.mid_cross_attn(x, context=None)
        intermediate_outputs['mid_crossattn' + suffix_key] = x
        x = self.mid_block2(x, t)
        intermediate_outputs['mid2' + suffix_key] = x



        # upsample
        up_counter = 0
        for block1, block2, cross_attn, upsample in self.ups:
            suffix_key = str(up_counter)
            x = torch.cat((x, h.pop()), dim=1)

            x = block1(x, t)
            intermediate_outputs['upsample_1' + suffix_key] = x

            x = block2(x, t)
            intermediate_outputs['upsample_2' + suffix_key] = x

            x = cross_attn(x, context)
            intermediate_outputs['up_cross_attn' + suffix_key] = x

            x = upsample(x)
            intermediate_outputs['upsample_fin' + suffix_key] = x

            up_counter += 1

        x = self.final_conv(x)
        intermediate_outputs['upsample_fin_conv'] = x

        return x, intermediate_outputs
