from __future__ import annotations

import math
import torch
import torch.nn as nn
from torch import distributed as dist

from dit_parallel.layers import (
    ColumnParallelLinear,
    RowParallelLinear,
    ring_attention,
    split_tensor_cp,
    gather_tensor_cp,
)
from dit_parallel.context import Context, setup_distributed


class Attention(nn.Module):
    def __init__(self, dim, num_heads, ctx: Context, attn_drop=0.0, proj_drop=0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        # old:
        # self.qkv = nn.Linear(dim, dim * 3)
        self.qkv = ColumnParallelLinear(dim, dim * 3, tp_dim=ctx.tp_pg.size(), ctx=ctx)
        self.attn_drop = nn.Dropout(attn_drop)
        # old:
        # self.proj = nn.Linear(dim, dim)
        self.proj = RowParallelLinear(
            dim,
            dim,
            tp_dim=ctx.tp_pg.size(),
            ctx=ctx,
            all_reduce_out=True,
        )
        self.proj_drop = nn.Dropout(proj_drop)
        self.ctx = ctx
        self.num_local_heads = self.num_heads // ctx.tp_pg.size()

    def forward(self, x):
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_local_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv
        dropout_p = self.attn_drop.p if self.training else 0.0
        ###
        # assumes x has already been split along the cp dimension
        x = ring_attention(q, k, v, dropout_p, self.ctx, backend="flash")
        ###
        x = x.transpose(1, 2).reshape(B, N, C // self.ctx.tp_pg.size())
        return self.proj_drop(self.proj(x))


def get_1d_sincos_pos_embed(d, pos):
    assert d % 2 == 0
    half = d // 2
    freqs = torch.exp(
        -math.log(10000.0) * torch.arange(half, device=pos.device) / half
    ).to(pos.dtype)
    args = pos.unsqueeze(1) * freqs.unsqueeze(0)
    return torch.cat([torch.sin(args), torch.cos(args)], dim=1)


def get_2d_sincos_pos_embed(d, gs):
    grid = torch.stack(
        torch.meshgrid(
            torch.arange(gs, device="cpu"),
            torch.arange(gs, device="cpu"),
            indexing="ij",
        ),
        0,
    ).reshape(2, -1)
    return torch.cat(
        [
            get_1d_sincos_pos_embed(d // 2, grid[0]),
            get_1d_sincos_pos_embed(d // 2, grid[1]),
        ],
        1,
    )


class PatchEmbed(nn.Module):
    def __init__(self, img_size, patch_size, in_ch, dim):
        super().__init__()
        self.patch_size = patch_size
        self.num_patches = (img_size // patch_size) ** 2
        self.proj = nn.Conv2d(in_ch, dim, patch_size, patch_size)

    def forward(self, x):
        x = self.proj(x)
        return x.flatten(2).transpose(1, 2)


class TimestepEmbedder(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(d, d), nn.SiLU(), nn.Linear(d, d))

    def forward(self, t):
        return self.mlp(get_1d_sincos_pos_embed(self.mlp[0].in_features, t))


class LabelEmbedder(nn.Module):
    def __init__(self, n_cls, d, p=0.1):
        super().__init__()
        self.table = nn.Embedding(n_cls + 1, d)
        self.n_cls = n_cls
        self.p = p

    def forward(self, y, train):
        if train and self.p:
            mask = torch.rand_like(y.float()) < self.p
            y = torch.where(mask, torch.full_like(y, self.n_cls), y)
        return self.table(y.long())


class Block(nn.Module):
    def __init__(
        self, d, heads, ctx: Context, mlp_ratio=4, attn_drop=0.0, proj_drop=0.0
    ):
        super().__init__()
        self.n1 = nn.LayerNorm(d)
        self.attn = Attention(d, heads, ctx, attn_drop, proj_drop)
        self.n2 = nn.LayerNorm(d)
        self.mlp = nn.Sequential(
            # old:
            # nn.Linear(d, d * mlp_ratio),
            ColumnParallelLinear(d, d * mlp_ratio, tp_dim=ctx.tp_pg.size(), ctx=ctx),
            nn.GELU(),
            # old:
            # nn.Linear(d * mlp_ratio, d),
            RowParallelLinear(
                d * mlp_ratio, d, tp_dim=ctx.tp_pg.size(), ctx=ctx, all_reduce_out=True
            ),
            nn.Dropout(proj_drop),
        )

    def forward(self, x):
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class ParallelDiT(nn.Module):
    def __init__(
        self,
        ctx: Context,
        img_size=4096,
        patch_size=8,
        in_c=4,
        d=512,
        depth=28,
        heads=16,
        n_cls=1000,
    ):
        super().__init__()
        self.ctx = ctx
        self.patch = PatchEmbed(img_size, patch_size, in_c, d)
        self.register_buffer(
            "pos",
            get_2d_sincos_pos_embed(d, int(math.sqrt(self.patch.num_patches))),
            False,
        )
        self.time = TimestepEmbedder(d)
        self.label = LabelEmbedder(n_cls, d, 0.1)
        self.blocks = nn.ModuleList([Block(d, heads, ctx=ctx) for _ in range(depth)])
        self.norm = nn.LayerNorm(d)
        self.head = nn.Linear(d, patch_size * patch_size * in_c)
        self.out_c = in_c
        self.patch_size = patch_size

    def _unpatch(self, x):
        B, N, _ = x.shape
        p = self.patch_size
        h = w = int(math.sqrt(N))
        return (
            x.view(B, h, w, self.out_c, p, p)
            .permute(0, 3, 1, 4, 2, 5)
            .reshape(B, self.out_c, h * p, w * p)
        )

    @torch.compile
    def forward(self, img, t, y):
        x = self.patch(img) + self.pos.unsqueeze(0)
        x = x + (self.time(t) + self.label(y, self.training)).unsqueeze(1)
        x = split_tensor_cp(x, dim=1, ctx=self.ctx)
        for blk in self.blocks:
            x = blk(x)
        x = gather_tensor_cp(x, dim=1, ctx=self.ctx)
        return self._unpatch(self.head(self.norm(x)))


if __name__ == "__main__":
    size = 2048
    ctx = setup_distributed(tp_dim=4, cp_dim=2)
    torch.cuda.set_device(dist.get_rank())
    device = f"cuda:{dist.get_rank()}"
    model = ParallelDiT(
        ctx=ctx, img_size=size, patch_size=8, d=512, depth=28, heads=16
    ).to(device, dtype=torch.bfloat16)

    # Warmup
    for _ in range(3):
        img = torch.randn(1, 4, size, size).to(device, dtype=torch.bfloat16)
        t = torch.randint(0, 1000, (1,)).to(device, dtype=torch.bfloat16)
        y = torch.randint(0, 1000, (1,)).to(device)
        with torch.no_grad():
            out = model(img, t, y)

    # Benchmark
    torch.cuda.synchronize()
    start_event = torch.cuda.Event(enable_timing=True)
    end_event = torch.cuda.Event(enable_timing=True)

    times = []
    for _ in range(10):
        img = torch.randn(1, 4, size, size).to(device, dtype=torch.bfloat16)
        t = torch.randint(0, 1000, (1,)).to(device, dtype=torch.bfloat16)
        y = torch.randint(0, 1000, (1,)).to(device)

        torch.cuda.synchronize()
        start_event.record()

        with torch.no_grad():
            out = model(img, t, y)

        end_event.record()
        torch.cuda.synchronize()
        times.append(start_event.elapsed_time(end_event))

    avg_time = sum(times) / len(times)
    print(f"Average inference time: {avg_time:.2f} ms")
    print(f"Throughput: {1000 / avg_time:.2f} images/s")
