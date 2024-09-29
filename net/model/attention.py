import torch
from torch import nn, einsum
import torch.nn.functional as F
from einops import rearrange, repeat
from timm.models.layers import DropPath
from functools import wraps

# adapted from https://github.com/1zb/3DShape2VecSet


def exists(val):
    return val is not None

def default(val, d):
    return val if exists(val) else d

def cache_fn(f):
    cache = None
    @wraps(f)
    def cached_fn(*args, _cache = True, **kwargs):
        if not _cache:
            return f(*args, **kwargs)
        nonlocal cache
        if cache is not None:
            return cache
        cache = f(*args, **kwargs)
        return cache
    return cached_fn


class PreNorm(nn.Module):
    def __init__(self, dim, fn, context_dim = None):
        super().__init__()
        self.fn = fn
        self.norm = nn.LayerNorm(dim)
        self.norm_context = nn.LayerNorm(context_dim) if exists(context_dim) else None

    def forward(self, x, **kwargs):
        x = self.norm(x)

        if exists(self.norm_context):
            context = kwargs['context']
            normed_context = self.norm_context(context)
            kwargs.update(context = normed_context)

        return self.fn(x, **kwargs)

class GEGLU(nn.Module):
    def forward(self, x):
        x, gates = x.chunk(2, dim = -1)
        return x * F.gelu(gates)

class FeedForward(nn.Module):
    def __init__(self, dim, mult = 4, drop_path_rate = 0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, dim * mult * 2),
            GEGLU(),
            nn.Linear(dim * mult, dim)
        )

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x):
        return self.drop_path(self.net(x))

class Attention(nn.Module):
    def __init__(self, query_dim, context_dim = None, heads = 8, dim_head = 64, drop_path_rate = 0.0):
        super().__init__()
        inner_dim = dim_head * heads
        context_dim = default(context_dim, query_dim)
        self.scale = dim_head ** -0.5
        self.heads = heads

        self.to_q = nn.Linear(query_dim, inner_dim, bias = False)
        self.to_kv = nn.Linear(context_dim, inner_dim * 2, bias = False)
        self.to_out = nn.Linear(inner_dim, query_dim)

        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0. else nn.Identity()

    def forward(self, x, context=None, mask=None):
        h = self.heads

        q = self.to_q(x)
        context = default(context, x)
        k, v = self.to_kv(context).chunk(2, dim = -1)

        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> (b h) n d', h = h), (q, k, v))

        sim = einsum('b i d, b j d -> b i j', q, k) * self.scale

        if exists(mask):
            mask = rearrange(mask, 'b ... -> b (...)')
            max_neg_value = -torch.finfo(sim.dtype).max
            mask = repeat(mask, 'b j -> (b h) () j', h = h)
            sim.masked_fill_(~mask, max_neg_value)

        attn = sim.softmax(dim = -1)

        out = einsum('b i j, b j d -> b i d', attn, v)
        out = rearrange(out, '(b h) n d -> b n (h d)', h = h)
        return self.drop_path(self.to_out(out))


class CrossAttention(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        self.cross_attend_blocks = nn.ModuleList([
            PreNorm(config.dim, Attention(config.dim, config.dim,
                                   heads=config.num_heads, dim_head=config.head_dim,
                                   drop_path_rate=config.drop_path_rate)),
            PreNorm(config.dim, FeedForward(config.dim, drop_path_rate=config.drop_path_rate))
        ])

    def forward(self, x, context=None, mask=None, decode=False):
        cross_attn, cross_ff = self.cross_attend_blocks

        if not decode:
            x = cross_attn(x, context=context, mask=mask) + x
            x = cross_ff(x) + x
        else:  # see AutoEncoder in https://github.com/1zb/3DShape2VecSet/blob/master/models_ae.py
            x = cross_attn(x, context=context, mask=mask)

        return x


class SelfAttention(nn.Module):

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config

        get_latent_attn = lambda: PreNorm(config.dim, Attention(config.dim, context_dim=None,
                                                                heads=config.num_heads, dim_head=config.head_dim,
                                                                drop_path_rate=config.drop_path_rate))
        get_latent_ff = lambda: PreNorm(config.dim, FeedForward(config.dim, drop_path_rate=config.drop_path_rate))
        get_latent_attn, get_latent_ff = map(cache_fn, (get_latent_attn, get_latent_ff))

        self.layers = nn.ModuleList([])
        cache_args = {'_cache': False}  # if True, weight tie layers
        for _ in range(config.depth):
            self.layers.append(nn.ModuleList([
                get_latent_attn(**cache_args),
                get_latent_ff(**cache_args)
            ]))

    def forward(self, x):
        for self_attn, self_ff in self.layers:
            x = self_attn(x) + x
            x = self_ff(x) + x

        return x


class ConditionalAttention(nn.Module):

    # alternating self-attention and cross-attention blocks

    def __init__(self, config) -> None:
        super().__init__()
        self.config = config
        self.sa_index = 0 if self.config.type == 'saca_attn' else 1  # casa

        self.layers = nn.ModuleList([])
        for i in range(self.config.depth):
            if i % 2 == self.sa_index:
                self.layers.append(SelfAttention(self.config.sa))
            else:
                self.layers.append(CrossAttention(self.config.ca))

    def forward(self, x, context):
        for i, layer in enumerate(self.layers):
            if i % 2 == self.sa_index:  # self-attention
                x = layer(x)
            else:  # cross-attention
                x = layer(x, context=context)
        return x
