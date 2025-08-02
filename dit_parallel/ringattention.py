from typing import Literal

import torch
from torch import distributed as dist
from torch.nn import functional as F

# flexattention
from torch.nn.attention.flex_attention import flex_attention
# fa3 
import flash_attn_interface
# fa2
from flash_attn import flash_attn_func

from dit_parallel.context import Context


flex_attention = torch.compile(flex_attention)



# custom op + fake implementation so fa3 works with torch.compile

@torch.library.custom_op("flash::torch_fa3", mutates_args=())
def torch_fa3(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    softmax_scale = q.shape[-1] ** -0.5
    out, softmax_lse, *rest = flash_attn_interface._flash_attn_forward(
        q,
        k,
        v,
        None, None,  # k_new, v_new
        None,  # qv
        None,  # out
        None, None, None,   # cu_seqlens_q/k/k_new
        None, None,   # seqused_q/k
        None, None,   # max_seqlen_q/k
        None, None, None,   # page_table, kv_batch_idx, leftpad_k,
        None, None, None,  # rotary_cos/sin, seqlens_rotary
        None, None, None,
        softmax_scale,
        causal=False,
        window_size=(-1, -1),
        attention_chunk=0,
        softcap=0.0,
        num_splits=1,
        pack_gqa=None,
        sm_margin=0,
    )

    return out, softmax_lse


@torch_fa3.register_fake
def _(q, k, v, **kwargs):
    # returns out, lse
    _out = torch.empty_like(q, dtype=torch.bfloat16)
    return _out, q.new_empty((q.size(0), q.size(2), q.size(1)), dtype=torch.float32)


def ring_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    dropout_p: float,
    ctx: Context,
    backend: Literal["fa2", "fa3", "flex"] = "fa3",
    dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Functional ring attention implementation.
    Assumes that q, k, v are already split along the context parallel dim."""

    q = q.to(dtype)
    k = k.to(dtype)
    v = v.to(dtype)

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

        if backend == "fa3":
            new_attn_out, new_lse = torch_fa3(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2))
            new_attn_out = new_attn_out.transpose(1, 2)
        elif backend == "fa2":
            if dtype == torch.float8_e4m3fn:
                raise NotImplementedError("float8_e4m3fn is not supported for fa2, choose fa3 or flex")
            new_attn_out, new_lse, _ = flash_attn_func(
                q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), return_attn_probs=True
            )
            new_attn_out = new_attn_out.transpose(1, 2)
        elif backend == "flex":
            new_attn_out, new_lse = flex_attention(q, k, v, return_lse=True)
        else:
            raise ValueError(
                f"Invalid backend: {backend}, should be one of ['fa2', 'fa3', 'flex']"
            )

        # todo: what dtype do I want to do the merge in
        # new_attn_out = new_attn_out.to(q.dtype)
        # new_lse = new_lse.to(q.dtype)
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


# some helper functions for context parallel

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
