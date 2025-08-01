import subprocess
import torch
from torch import distributed as dist

from dit_parallel.context import setup_distributed
from parallel_model import ParallelDiT


def print0(s: str):
    if dist.get_rank() == 0:
        print(s)


def profile():
    size = 2048
    TP_DIM = 2
    CP_DIM = 4
    assert TP_DIM * CP_DIM == torch.cuda.device_count()
    ctx = setup_distributed(tp_dim=TP_DIM, cp_dim=CP_DIM)
    device = f"cuda:{dist.get_rank()}"
    model = ParallelDiT(
        ctx=ctx, img_size=size, patch_size=8, d=512, depth=28, heads=16
    ).to(device, dtype=torch.bfloat16)

    torch.manual_seed(1234)
    print0(f"Starting warmup for {TP_DIM}x{CP_DIM}x{dist.get_rank()}")
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

    with torch.profiler.profile() as prof:
        with torch.no_grad():
            model(img, t, y)

    from datetime import datetime

    timestamp = datetime.now().strftime("%H%M")
    profile_name = f"{timestamp}_dit_tp{TP_DIM}_cp{CP_DIM}_rank{dist.get_rank()}"

    prof.export_chrome_trace(f"/tmp/traces/{profile_name}.json")
    subprocess.run(
        [
            "tar",
            "-czvf",
            f"/tmp/traces/{profile_name}.tar.gz",
            f"/tmp/traces/{profile_name}.json",
        ]
    )


if __name__ == "__main__":
    profile()
