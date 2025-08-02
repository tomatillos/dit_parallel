import torch
from torch import nn
from torch import distributed as dist

from dit_parallel.context import Context


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
        # remember to rescale bias when loading weights if all_reduce_out is True
        self.all_reduce_out = all_reduce_out
        assert self.ctx.tp_pg.size() == tp_dim
        if with_bias and not all_reduce_out:
            raise NotImplementedError

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = super().forward(x)
        if self.all_reduce_out:
            dist.all_reduce(out, op=dist.ReduceOp.SUM, group=self.ctx.tp_pg)
        return out
