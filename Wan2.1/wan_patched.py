import torch
from torch import nn
from torch import distributed as dist
from wan.modules.model import WanRMSNorm, rope_apply

from dit_parallel.ringattention import ring_attention, split_tensor_cp, gather_tensor_cp
from dit_parallel.tp import ColumnParallelLinear, RowParallelLinear

from dit_parallel.context import setup_distributed


class PatchedWanSelfAttention(nn.Module):

    def __init__(self,
                 dim,
                 num_heads,
                 window_size=(-1, -1),
                 qk_norm=True,
                 eps=1e-6,
                 ctx=None):
        assert dim % num_heads == 0
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.window_size = window_size
        self.qk_norm = qk_norm
        self.eps = eps

        # layers
        # self.q = nn.Linear(dim, dim)
        # self.k = nn.Linear(dim, dim)
        # self.v = nn.Linear(dim, dim)
        # self.o = nn.Linear(dim, dim)
        self.q = ColumnParallelLinear(dim, dim, tp_dim=ctx.tp_pg.size(), ctx=ctx, all_gather_out=False, with_bias=True)
        self.k = ColumnParallelLinear(dim, dim, tp_dim=ctx.tp_pg.size(), ctx=ctx, all_gather_out=False, with_bias=True)
        self.v = ColumnParallelLinear(dim, dim, tp_dim=ctx.tp_pg.size(), ctx=ctx, all_gather_out=False, with_bias=True)
        self.o = RowParallelLinear(dim, dim, tp_dim=ctx.tp_pg.size(), ctx=ctx, all_reduce_out=True, with_bias=True)

        self.norm_q = WanRMSNorm(dim // ctx.tp_pg.size(), eps=eps) if qk_norm else nn.Identity()
        self.norm_k = WanRMSNorm(dim // ctx.tp_pg.size(), eps=eps) if qk_norm else nn.Identity()

        self.ctx = ctx

    def forward(self, x, seq_lens, grid_sizes, freqs):
        r"""
        Args:
            x(Tensor): Shape [B, L, num_heads, C / num_heads]
            seq_lens(Tensor): Shape [B]
            grid_sizes(Tensor): Shape [B, 3], the second dimension contains (F, H, W)
            freqs(Tensor): Rope freqs, shape [1024, C / num_heads / 2]
        """
        b, s, n, d = *x.shape[:2], self.num_heads, self.head_dim

        n = n // self.ctx.tp_pg.size()

        # query, key, value function
        def qkv_fn(x):
            q = self.norm_q(self.q(x)).view(b, s, n, d)
            k = self.norm_k(self.k(x)).view(b, s, n, d)
            v = self.v(x).view(b, s, n, d)
            return q, k, v

        q, k, v = qkv_fn(x)
        q = rope_apply(q, grid_sizes, freqs)
        k = rope_apply(k, grid_sizes, freqs)

        q = split_tensor_cp(q, dim=1, ctx=self.ctx)
        k = split_tensor_cp(k, dim=1, ctx=self.ctx)
        v = split_tensor_cp(v, dim=1, ctx=self.ctx)
        x = ring_attention(
            q=q.transpose(1,2),
            k=k.transpose(1,2),
            v=v.transpose(1,2),
            dropout_p=0.0,
            ctx=self.ctx,
            backend="fa3",
            dtype=torch.bfloat16,
        )
        x = x.transpose(1,2)
        # x = flash_attention(
            # q=rope_apply(q, grid_sizes, freqs),
            # k=rope_apply(k, grid_sizes, freqs),
            # v=v,
            # k_lens=seq_lens,
            # window_size=self.window_size)

        # output
        x = x.flatten(2)
        x = self.o(x)
        x = gather_tensor_cp(x, dim=1, ctx=self.ctx)
        return x


class PatchedWanT2VCrossAttention(PatchedWanSelfAttention):

    def forward(self, x, context, context_lens):
        r"""
        Args:
            x(Tensor): Shape [B, L1, C]
            context(Tensor): Shape [B, L2, C]
            context_lens(Tensor): Shape [B]
        """
        b, n, d = x.size(0), self.num_heads, self.head_dim
        n = n // self.ctx.tp_pg.size()

        x = split_tensor_cp(x, dim=1, ctx=self.ctx)
        # compute query, key, value
        q = self.norm_q(self.q(x)).view(b, -1, n, d)
        k = self.norm_k(self.k(context)).view(b, -1, n, d)
        v = self.v(context).view(b, -1, n, d)

        # compute attention
        # x = flash_attention(q, k, v, k_lens=context_lens)
        x = ring_attention(
            q=q.transpose(1,2),
            k=k.transpose(1,2),
            v=v.transpose(1,2),
            dropout_p=0.0,
            ctx=self.ctx,
            backend="fa3",
            dtype=torch.bfloat16,
        )
        x = x.transpose(1,2)

        # output
        x = x.flatten(2)
        x = self.o(x)
        x = gather_tensor_cp(x, dim=1, ctx=self.ctx)
        return x


def print0(s: str):
    if dist.get_rank() == 0:
        print(s)


@torch.no_grad()
def parallelize_wan(wan_model):
    ctx = setup_distributed(tp_dim=2, cp_dim=4)

    tp_size = ctx.tp_pg.size()
    tp_rank = ctx.tp_pg.rank()

    nheads = wan_model.num_heads

    def copy_qkv_col_parallel_linear(old_layer, new_layer, bias=True):
        old_weight = old_layer.weight
        d_out, d_in = old_weight.shape
        old_weight_t = old_weight.transpose(0, 1)
        old_weight_t_split  = old_weight_t.view(d_in, nheads, -1)
        old_weight_t_chunked = torch.chunk(old_weight_t_split, dim=1, chunks=tp_size)[tp_rank].view(d_in, d_out//tp_size)
        new_weight = old_weight_t_chunked.transpose(0, 1)
        new_layer.weight.copy_(new_weight)
        if bias:
            new_layer.bias.copy_(torch.chunk(old_layer.bias, dim=0, chunks=tp_size)[tp_rank])

    def copy_col_parallel_linear(old_layer, new_layer, bias=True):
        new_layer.weight.copy_(torch.chunk(old_layer.weight, dim=0, chunks=tp_size)[tp_rank])
        if bias:
            new_layer.bias.copy_(torch.chunk(old_layer.bias, dim=0, chunks=tp_size)[tp_rank])

    def copy_row_parallel_linear(old_layer, new_layer):
        new_layer.weight.copy_(torch.chunk(old_layer.weight, dim=1, chunks=tp_size)[tp_rank])
        new_layer.bias.copy_(old_layer.bias / tp_size)

    for block in wan_model.blocks:
        # attn
        new_block_attn = PatchedWanSelfAttention(
            dim=block.dim,
            num_heads=block.num_heads,
            window_size=block.window_size,
            qk_norm=block.qk_norm,
            eps=block.eps,
            ctx=ctx,
        )
        copy_qkv_col_parallel_linear(block.self_attn.q, new_block_attn.q)
        copy_qkv_col_parallel_linear(block.self_attn.k, new_block_attn.k)
        copy_qkv_col_parallel_linear(block.self_attn.v, new_block_attn.v)
        copy_row_parallel_linear(block.self_attn.o, new_block_attn.o)
        copy_col_parallel_linear(block.self_attn.norm_q, new_block_attn.norm_q, bias=False)
        copy_col_parallel_linear(block.self_attn.norm_k, new_block_attn.norm_k, bias=False)
        block.self_attn = new_block_attn

        new_block_cross_attn = PatchedWanT2VCrossAttention(
            dim=block.dim,
            num_heads=block.num_heads,
            window_size=block.window_size,
            qk_norm=block.qk_norm,
            eps=block.eps,
            ctx=ctx,
        )
        copy_qkv_col_parallel_linear(block.cross_attn.q, new_block_cross_attn.q)
        copy_qkv_col_parallel_linear(block.cross_attn.k, new_block_cross_attn.k)
        copy_qkv_col_parallel_linear(block.cross_attn.v, new_block_cross_attn.v)
        copy_row_parallel_linear(block.cross_attn.o, new_block_cross_attn.o)
        copy_col_parallel_linear(block.cross_attn.norm_q, new_block_cross_attn.norm_q, bias=False)
        copy_col_parallel_linear(block.cross_attn.norm_k, new_block_cross_attn.norm_k, bias=False)
        block.cross_attn = new_block_cross_attn

        # ffn
        new_block_ffn = nn.Sequential(
            ColumnParallelLinear(block.dim, block.ffn_dim, tp_dim=tp_size, ctx=ctx, all_gather_out=False, with_bias=True),
            nn.GELU(approximate='tanh'),
            RowParallelLinear(block.ffn_dim, block.dim, tp_dim=tp_size, ctx=ctx, all_reduce_out=True, with_bias=True)
        )

        copy_col_parallel_linear(block.ffn[0], new_block_ffn[0])
        copy_row_parallel_linear(block.ffn[2], new_block_ffn[2])

        block.ffn = new_block_ffn

    return wan_model

