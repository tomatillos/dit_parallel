import torch
from torch import nn
from torch import distributed as dist
from torch.nn import functional as F
from typing import Literal
from torch.nn.attention.flex_attention import flex_attention
from flash_attn import flash_attn_func

from dit_parallel.context import Context

flex_attention = torch.compile(flex_attention)


class ColumnParallelLinear(nn.Linear):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        tp_dim: int,
        ctx: Context,
        all_gather_out: bool = False,
        with_bias: bool = True,
    ):

        assert d_out % tp_dim == 0
        local_d_out = d_out // tp_dim
        super().__init__(in_features=d_in, out_features=local_d_out, bias=with_bias)
        self.tp_dim = tp_dim

        self.all_gather_out = all_gather_out
        self.ctx = ctx
        assert self.ctx.tp_pg.size() == tp_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        if self.all_gather_out:
            out_tensors = [torch.empty_like(out) for _ in range(self.tp_dim)]
            dist.all_gather(out_tensors, out, group=self.ctx.tp_pg)
            out = torch.cat(out_tensors, dim=-1).to(out.device)
        return out


class RowParallelLinear(nn.Linear):
    def __init__(
        self,
        d_in: int,
        d_out: int,
        tp_dim: int,
        ctx: Context,
        all_reduce_out: bool = True,
        with_bias: bool = True,
    ):
        assert d_in % tp_dim == 0
        local_d_in = d_in // tp_dim
        super().__init__(in_features=local_d_in, out_features=d_out, bias=with_bias)
        self.ctx = ctx
        # remember to rescale the bias when loading weights
        self.all_reduce_out = all_reduce_out
        assert self.ctx.tp_pg.size() == tp_dim
        if with_bias and not all_reduce_out:
            raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        if self.all_reduce_out:
            dist.all_reduce(out, op=dist.ReduceOp.SUM, group=self.ctx.tp_pg)
        return out


def ring_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    ctx: Context,
    backend: Literal["flash", "flex"] = "flash",
) -> torch.Tensor:
    """Functional ring attention implementation."""

    k = k.contiguous()
    v = v.contiguous()
    k_buf = torch.empty_like(k)
    v_buf = torch.empty_like(v)

    cp_rank = dist.get_rank(ctx.cp_pg)
    cp_dim = ctx.cp_pg.size()

    next_loc = (cp_rank + 1) % cp_dim
    prev_loc = (cp_rank - 1) % cp_dim

    attn_out, lse = None, None

    for step in range(cp_dim):
        # send/recv the k and v to the adjacent ranks
        if step < cp_dim - 1:
            send_k = dist.P2POp(dist.isend, k, group=ctx.cp_pg, group_peer=next_loc)
            recv_k = dist.P2POp(dist.irecv, k_buf, group=ctx.cp_pg, group_peer=prev_loc)
            send_v = dist.P2POp(dist.isend, v, group=ctx.cp_pg, group_peer=next_loc)
            recv_v = dist.P2POp(dist.irecv, v_buf, group=ctx.cp_pg, group_peer=prev_loc)
            reqs = dist.batch_isend_irecv([send_k, recv_k, send_v, recv_v])

        if backend == "flash":
            new_attn_out, new_lse, _ = flash_attn_func(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), return_attn_probs=True
            )
            new_attn_out = new_attn_out.transpose(1,2)
        elif backend == "flex":
            new_attn_out, new_lse = flex_attention(q, k, v, return_lse=True)
        else:
            raise ValueError(
                f"Invalid backend: {backend}, should be one of ['flash', 'flex']"
            )

        new_attn_out = new_attn_out.to(q.dtype)
        new_lse = new_lse.to(q.dtype)
        new_lse = new_lse.unsqueeze(-1)
        # merge the two attention states
        if step == 0:
            lse = new_lse
            attn_out = new_attn_out
        else:
            # slightly unintuitive merge formula, torch.dist does it this way:
            # https://github.com/pytorch/pytorch/blob/3967dbedf4bd7ecb8bfae93e5b8ec78e8f523b9a/torch/distributed/tensor/experimental/_attention.py#L201
            # cf https://github.com/zhuzilin/ring-flash-attention/pull/34#issuecomment-2076126795
            attn_out = attn_out - F.sigmoid(new_lse - lse) * (attn_out - new_attn_out)
            lse = lse - F.logsigmoid(lse - new_lse)

        if step < cp_dim - 1:
            for req in reqs:
                req.wait()

            # now buf is the next one, so swap
            k, k_buf = k_buf, k
            v, v_buf = v_buf, v


    return attn_out.to(k.dtype)


def split_tensor_cp(x: torch.Tensor, dim: int, ctx: Context) -> torch.Tensor:
    """Helper function to split a tensor along the context parallel dimension."""
    cp_rank = dist.get_rank(ctx.cp_pg)
    cp_dim = ctx.cp_pg.size()
    x = x.chunk(cp_dim, dim=dim)[cp_rank]
    return x


def gather_tensor_cp(x: torch.Tensor, dim: int, ctx: Context) -> torch.Tensor:
    """Helper function to gather a tensor along the context parallel dimension."""
    cp_rank = dist.get_rank(ctx.cp_pg)
    cp_dim = ctx.cp_pg.size()
    x = x.contiguous()
    x_tensors = [torch.empty_like(x) for _ in range(cp_dim)]
    x_tensors[cp_rank] = x
    dist.all_gather(x_tensors, x, group=ctx.cp_pg)
    return torch.cat(x_tensors, dim=dim)
