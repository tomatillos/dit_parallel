import os
import subprocess

import torch
from torch import distributed as dist
from torchao.quantization import quantize_, Float8DynamicActivationFloat8WeightConfig

from dit_parallel.context import setup_distributed
from dit_parallel.models.parallel_dit import ParallelDiT


def print0(s: str):
    if dist.get_rank() == 0:
        print(s)


def profile_model(tp_dim, cp_dim, torch_compile=True, quantize_fp8=False):

    if tp_dim * cp_dim != torch.cuda.device_count():
        print(f"TP_DIM * CP_DIM ({tp_dim * cp_dim}) != torch.cuda.device_count() ({torch.cuda.device_count()})")

    ctx = setup_distributed(tp_dim=tp_dim, cp_dim=cp_dim)
    device = f"cuda:{dist.get_rank()}"
    torch.cuda.set_device(dist.get_rank())

    size = 2048
    attn_dtype = torch.float8_e4m3fn if quantize_fp8 else torch.bfloat16
    model = ParallelDiT(
        ctx=ctx, img_size=size, patch_size=8, d=512, depth=28, heads=16, attn_dtype=attn_dtype
    ).to(device, dtype=torch.bfloat16)

    if quantize_fp8:
        print0("Quantizing model")
        quantize_(model.blocks, Float8DynamicActivationFloat8WeightConfig())

    if torch_compile:
        model.forward = torch.compile(model.forward)

    torch.manual_seed(1234)
    print0(f"Starting warmup for {tp_dim}x{cp_dim}x{dist.get_rank()} with torch.compile={torch_compile} and quantize_fp8={quantize_fp8}")
    # Warmup
    for _ in range(3):
        img = torch.randn(1, 4, size, size).to(device, dtype=torch.bfloat16)
        t = torch.randint(0, 1000, (1,)).to(device, dtype=torch.bfloat16)
        y = torch.randint(0, 1000, (1,)).to(device)
        with torch.no_grad():
            out = model(img, t, y)

    print0("Finished warmup")

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
    print0(f"Average inference time: {avg_time:.2f} ms")
    print0(f"Throughput: {1000 / avg_time:.2f} images/s")

    torch.cuda.synchronize()

    with torch.profiler.profile() as prof:
        with torch.no_grad():
            model(img, t, y)

    from datetime import datetime

    timestamp = datetime.now().strftime("%H%M")
    profile_name = f"{timestamp}_dit_tp{tp_dim}_cp{cp_dim}_rank{dist.get_rank()}"

    os.makedirs("/tmp/traces", exist_ok=True)
    prof.export_chrome_trace(f"/tmp/traces/{profile_name}.json")
    subprocess.run(
        [
            "tar",
            "-czvf",
            f"/tmp/traces/{profile_name}.tar.gz",
            f"/tmp/traces/{profile_name}.json",
        ],
        stderr=subprocess.DEVNULL,
    )

    dist.destroy_process_group()


if __name__ == "__main__":
    profile_model(tp_dim=2, cp_dim=4, torch_compile=True, quantize_fp8=False)
