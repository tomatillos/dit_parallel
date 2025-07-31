import torch
from torch import nn
from torch import distributed as dist

from flash_attn import flash_attn_func

class ColumnParallelLinear(nn.Module):
    def __init__(self, d_in: int, d_out: int, tp_dim: int, dtype: torch.dtype, device: torch.device, all_gather_out: bool=False):
        super().__init__()
        assert d_out % tp_dim == 0
        local_d_out = d_out // tp_dim
        self.tp_size = tp_dim
        self.w = nn.Parameter(torch.empty(local_d_out, d_in, dtype=dtype, device=device))
        self.all_gather_out = all_gather_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x @ self.w.T
        if self.all_gather_out:
            out_tensors = [torch.empty_like(out) for _ in range(self.tp_size)]
            dist.all_gather(out_tensors, out)
            out = torch.cat(out_tensors, dim=-1).to(out.device)
        return out


class RowParallelLinear(nn.Module):
    def __init__(self, d_in: int, d_out: int, tp_dim: int, dtype: torch.dtype, device: torch.device, all_reduce_out: bool=True):
        super().__init__()
        assert d_in % tp_dim == 0
        local_d_in = d_in // tp_dim
        self.w = nn.Parameter(torch.empty(d_out, local_d_in, dtype=dtype, device=device))
        self.all_reduce_out = all_reduce_out

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = x @ self.w.T
        if self.all_reduce_out:
            dist.all_reduce(out, op=dist.ReduceOp.SUM)
        return out


class RingAttention(nn.Module):

    def __init__(self, dim: int, num_heads: int, cp_dim: int, attn_drop: float = 0.0, proj_drop: float = 0.0):
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.cp_dim = cp_dim
        def get_rank():
            raise NotImplementedError
        self.rank = get_rank()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # assume that x is **already** chunked on sequence dim (N)
        B, N, C = x.shape
        qkv = (
            self.qkv(x)
            .reshape(B, N, 3, self.num_heads, self.head_dim)
            .permute(2, 0, 3, 1, 4)
        )
        q, k, v = qkv
        # buffers to pass around the kv's
        k_buf = torch.empty_like(k)
        v_buf = torch.empty_like(v)

        next_loc = (self.rank + 1) % self.cp_dim
        prev_loc = (self.rank - 1) % self.cp_dim

        attn_out, lse = None, None

        for step in range(self.cp_dim):
            # send/recv the k and v to the next rank

            # todo: batch isend irecv
            work = []
            work.append(dist.isend(k, next_loc))
            work.append(dist.irecv(k_buf, prev_loc))
            work.append(dist.isend(v, next_loc))
            work.append(dist.irecv(v_buf, prev_loc))

            new_attn_out, new_lse, _ = (
                flash_attn_func(
                    q,
                    k,
                    v,
                    dropout_p=self.attn_drop.p if self.training else 0.0,
                    return_attn_probs=True,
                )
            )
            # and then do the recombine
            # todo: there are more stable ways to do the merge 
            if lse:
                attn_out = attn_out * torch.exp(lse) + new_attn_out * torch.exp(new_lse)
                denom = torch.exp(lse) + torch.exp(new_lse)
                lse = torch.log(denom)
                attn_out = attn_out / denom
            else:
                lse = new_lse
                attn_out = new_attn_out

            for handle in work:
                handle.wait()

            # now buf is the next one, so swap
            k, k_buf = k_buf, k
            v, v_buf = v_buf, v

        x = attn_out.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))

