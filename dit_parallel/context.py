from dataclasses import dataclass
import os

from torch import distributed as dist
from torch.distributed.device_mesh import init_device_mesh


@dataclass
class Context:
    tp_pg: dist.ProcessGroup
    cp_pg: dist.ProcessGroup


def setup_distributed(tp_dim: int, cp_dim: int) -> Context:
    if not dist.is_initialized():
        dist.init_process_group(
            backend="nccl",
            init_method="env://",
            world_size=int(os.environ.get("WORLD_SIZE", 1)),
            rank=int(os.environ.get("RANK", 0)),
        )

    mesh = init_device_mesh("cuda", (tp_dim, cp_dim), mesh_dim_names=("tp", "cp"))

    tp_pg = mesh["tp"].get_group()
    cp_pg = mesh["cp"].get_group()

    ctx = Context(tp_pg, cp_pg)
    return ctx
