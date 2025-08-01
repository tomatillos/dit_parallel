import torch
import pickle
import logging
from torch import distributed as dist

from dit_parallel.models.parallel_model import ParallelDiT
from dit_parallel.context import setup_distributed


def load_ref_weights_to_parallel_model(parallel_model, ref_state_dict):
    parallel_state_dict = parallel_model.state_dict()

    for name, param in ref_state_dict.items():
        # col parallel
        if name.endswith(".qkv.weight") or (
            name.endswith(".weight") and "mlp.0" in name
        ):
            parallel_name = name
            if parallel_name in parallel_state_dict:
                local_param = param.chunk(parallel_model.ctx.tp_pg.size(), dim=0)[
                    parallel_model.ctx.tp_pg.rank()
                ]
                parallel_state_dict[parallel_name].copy_(local_param)
        elif name.endswith(".qkv.bias") or (name.endswith(".bias") and "mlp.0" in name):
            parallel_name = name
            if parallel_name in parallel_state_dict:
                local_param = param.chunk(parallel_model.ctx.tp_pg.size(), dim=0)[
                    parallel_model.ctx.tp_pg.rank()
                ]
                parallel_state_dict[parallel_name].copy_(local_param)
        # row parallel
        elif name.endswith(".proj.weight") or (
            name.endswith(".weight") and "mlp.2" in name
        ):
            parallel_name = name
            if parallel_name in parallel_state_dict:
                local_param = param.chunk(parallel_model.ctx.tp_pg.size(), dim=1)[
                    parallel_model.ctx.tp_pg.rank()
                ]
                parallel_state_dict[parallel_name].copy_(local_param)
        else:
            parallel_state_dict[name].copy_(param)


def print0(s: str):
    if dist.get_rank() == 0:
        print(s)


def main():
    tp_dim = 4
    cp_dim = 2
    logging.info(f"Testing with tp_dim={tp_dim} and cp_dim={cp_dim}")
    ctx = setup_distributed(tp_dim=tp_dim, cp_dim=cp_dim)
    with open("test_outputs/ref_model_data.pkl", "rb") as f:
        ref_data = pickle.load(f)

    config = ref_data["model_config"]
    ref_state_dict = ref_data["model_state_dict"]
    inputs = ref_data["inputs"]
    ref_output = ref_data["ref_output"]

    device = f"cuda:{dist.get_rank()}"
    dtype = torch.bfloat16

    parallel_model = ParallelDiT(
        ctx=ctx,
        img_size=config["img_size"],
        patch_size=config["patch_size"],
        in_c=config["in_c"],
        d=config["d"],
        depth=config["depth"],
        heads=config["heads"],
        n_cls=config["n_cls"],
    ).to(device, dtype=dtype)

    parallel_model.eval()

    load_ref_weights_to_parallel_model(parallel_model, ref_state_dict)
    logging.info("Loaded weights successfully")

    img = inputs["img"].to(device, dtype=dtype)
    t = inputs["t"].to(device, dtype=dtype)
    y = inputs["y"].to(device)

    with torch.no_grad():
        parallel_output = parallel_model(img, t, y)

    print0("\n" + "=" * 50)
    print0("COMPARISON RESULTS")
    print0("=" * 50)

    ref_output = ref_output.to(device, dtype=dtype)

    abs_diff = torch.abs(parallel_output - ref_output)
    rel_diff = abs_diff / (torch.abs(ref_output) + 1e-8)

    print0(f"Absolute difference stats:")
    print0(f"  Mean: {abs_diff.mean().item():.8f}")
    print0(f"  Max:  {abs_diff.max().item():.8f}")
    print0(f"  Std:  {abs_diff.std().item():.8f}")

    print0(f"\nRelative difference stats:")
    print0(f"  Mean: {rel_diff.mean().item():.8f}")
    print0(f"  Max:  {rel_diff.max().item():.8f}")
    print0(f"  Std:  {rel_diff.std().item():.8f}")

    rtol = 1e-4
    atol = 1e-5
    are_close = torch.allclose(parallel_output, ref_output, rtol=rtol, atol=atol)

    print0(f"\nOutputs are close (rtol={rtol}, atol={atol}): {are_close}")

    if are_close:
        print0("SUCCESS: Outputs match!")
    else:
        print0("FAILED: Outputs differ beyond tolerance")

    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
