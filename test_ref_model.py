import pickle
import os

import torch

from dit_parallel.models.ref_model import DiT


def main():
    torch.manual_seed(1234)

    device = "cuda"
    dtype = torch.bfloat16

    size = 2048
    ref_model = DiT(
        img_size=size,
        patch_size=8,
        d=512,
        depth=28,
        heads=16,
    ).to(device, dtype=dtype)

    ref_model.eval()

    torch.manual_seed(1234)
    img = torch.randn(1, 4, size, size, dtype=dtype, device=device)
    t = torch.randint(0, 1000, (1,)).to("cuda", dtype=torch.bfloat16)
    y = torch.randint(0, 1000, (1,)).to("cuda")

    with torch.no_grad():
        ref_output = ref_model(img, t, y)

    print(f"Reference output shape: {ref_output.shape}")
    print(
        f"Reference output stats: mean={ref_output.mean().item():.6f}, std={ref_output.std().item():.6f}"
    )
    print(
        f"Reference output range: [{ref_output.min().item():.6f}, {ref_output.max().item():.6f}]"
    )

    data = {
        "model_state_dict": ref_model.state_dict(),
        "inputs": {"img": img.cpu(), "t": t.cpu(), "y": y.cpu()},
        "ref_output": ref_output.cpu(),
        "device": device,
        "dtype": str(dtype),
    }

    os.makedirs("test_outputs", exist_ok=True)
    with open("test_outputs/ref_model_data.pkl", "wb") as f:
        pickle.dump(data, f)


if __name__ == "__main__":
    main()
