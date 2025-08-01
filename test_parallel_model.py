import torch
import pickle
from torch import distributed as dist

from dit_parallel.models.parallel_model import ParallelDiT
from dit_parallel.context import setup_distributed


def load_ref_weights_to_parallel_model(parallel_model, ref_state_dict):
    parallel_state_dict = parallel_model.state_dict()
    nheads = 16 # todo: don't hardcode this
    tp_size = parallel_model.ctx.tp_pg.size()
    tp_rank = parallel_model.ctx.tp_pg.rank()
    for name, param in ref_state_dict.items():
        if "blocks" not in name:
            local_param = param
        # column parallel
        elif name.endswith(".qkv.weight"):
            param_t = param.transpose(0, 1)
            d_in, d_out = param_t.shape
            param_t = param_t.reshape(d_in, 3, nheads, -1)
            local_param_t = param_t.chunk(tp_size, dim=2)[tp_rank]
            local_param_t = local_param_t.reshape(d_in, -1)
            local_param = local_param_t.transpose(0, 1)
        elif name.endswith(".qkv.bias"):
            param = param.view(3, nheads, -1)
            local_param = param.chunk(tp_size, dim=1)[tp_rank]
            local_param = local_param.reshape(-1)
        elif name.endswith("mlp.0.weight"):
            local_param = param.chunk(tp_size, dim=0)[tp_rank]
        elif name.endswith("mlp.0.bias"):
            local_param = param.chunk(tp_size, dim=0)[tp_rank]
        # row parallel
        elif name.endswith(".proj.weight") or name.endswith("mlp.2.weight"):
            local_param = param.chunk(tp_size, dim=1)[tp_rank]
        elif name.endswith("mlp.2.bias") or name.endswith(".proj.bias"):
            # rescale bias since we all-reduce
            local_param = param / tp_size
        else:
            local_param = param

        parallel_state_dict[name].copy_(local_param)


def print0(s: str):
    if dist.get_rank() == 0:
        print(s)


def main():
    tp_dim = 2
    cp_dim = 4
    ctx = setup_distributed(tp_dim=tp_dim, cp_dim=cp_dim)
    with open("test_outputs/ref_model_data.pkl", "rb") as f:
        ref_data = pickle.load(f)

    ref_output = ref_data["ref_output"]

    device = f"cuda:{dist.get_rank()}"
    dtype = torch.bfloat16

    size = 2048
    parallel_model = ParallelDiT(
        ctx=ctx, img_size=size, patch_size=8, d=512, depth=28, heads=16
    ).to(device, dtype=dtype)

    parallel_model.eval()

    load_ref_weights_to_parallel_model(parallel_model, ref_data["model_state_dict"])
    print0("Loaded weights successfully")

    torch.manual_seed(1234)
    img = torch.randn(1, 4, size, size, dtype=dtype, device=device)
    t = torch.randint(0, 1000, (1,)).to(device, dtype=torch.bfloat16)
    y = torch.randint(0, 1000, (1,)).to(device)

    with torch.no_grad():
        parallel_output = parallel_model(img, t, y)

    print0("\n" + "=" * 50)
    print0("COMPARISON RESULTS")
    print0("=" * 50)

    ref_output = ref_output.to(device, dtype=dtype)

    abs_diff = torch.abs(parallel_output - ref_output)

    print0(f"Mean: {abs_diff.mean().item():.8f}")
    print0(f"Max:  {abs_diff.max().item():.8f}")
    print0(f"Std:  {abs_diff.std().item():.8f}")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
