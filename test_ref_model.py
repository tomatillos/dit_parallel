import pickle
import os
import logging

import torch

from dit_parallel.models.ref_model import DiT


def main():
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    img_size = 4096
    patch_size = 8
    in_c = 4
    d = 512
    depth = 28
    heads = 16
    n_cls = 1000

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32

    ref_model = DiT(
        img_size=img_size,
        patch_size=patch_size,
        in_c=in_c,
        d=d,
        depth=depth,
        heads=heads,
        n_cls=n_cls,
    ).to(device, dtype=dtype)

    ref_model.eval()

    torch.manual_seed(1234)
    img = torch.randn(1, in_c, img_size, img_size, dtype=dtype, device=device)
    t = torch.tensor([500], dtype=dtype, device=device)
    y = torch.tensor([123], dtype=torch.long, device=device)

    with torch.no_grad():
        ref_output = ref_model(img, t, y)

    print(f"Reference output shape: {ref_output.shape}")
    print(
        f"Reference output stats: mean={ref_output.mean().item():.6f}, std={ref_output.std().item():.6f}"
    )
    print(
        f"Reference output range: [{ref_output.min().item():.6f}, {ref_output.max().item():.6f}]"
    )

    # Save model state dict, inputs, and output
    data = {
        "model_state_dict": ref_model.state_dict(),
        "model_config": {
            "img_size": img_size,
            "patch_size": patch_size,
            "in_c": in_c,
            "d": d,
            "depth": depth,
            "heads": heads,
            "n_cls": n_cls,
        },
        "inputs": {"img": img.cpu(), "t": t.cpu(), "y": y.cpu()},
        "ref_output": ref_output.cpu(),
        "device": device,
        "dtype": str(dtype),
    }

    os.makedirs("test_outputs", exist_ok=True)
    with open("test_outputs/ref_model_data.pkl", "wb") as f:
        pickle.dump(data, f)

    logging.info("Saved reference model data to test_outputs/ref_model_data.pkl")
    logging.info("Reference model test completed successfully!")


if __name__ == "__main__":
    main()
