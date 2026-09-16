"""Check imports and a tiny CUDA forward/backward pass without downloading models."""
import json
import os
import platform
import subprocess
from importlib.metadata import version

import torch
import torchvision
import yaml
import pandas
import imageio.v3
import trimesh
import onnxruntime
from rembg import remove
from diffusers import StableDiffusionXLPipeline
from transformers import CLIPTextModel, CLIPTokenizer
from gsplat import rasterization


def main():
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU is unavailable")
    device = "cuda"
    means = torch.tensor([[0., 0., 3.]], device=device, requires_grad=True)
    quats = torch.tensor([[1., 0., 0., 0.]], device=device)
    scales = torch.full((1, 3), .1, device=device)
    opacities = torch.tensor([.8], device=device)
    colors = torch.tensor([[1., .3, .1]], device=device)
    views = torch.eye(4, device=device)[None]
    intrinsic = torch.tensor([[[32., 0., 16.], [0., 32., 16.], [0., 0., 1.]]], device=device)
    rgb, alpha, _ = rasterization(means, quats, scales, opacities, colors,
                                  views, intrinsic, 32, 32, packed=False)
    (rgb.sum() + alpha.sum()).backward()
    torch.cuda.synchronize()
    assert rgb.shape == (1, 32, 32, 3)
    assert alpha.sum().item() > 0
    assert torch.isfinite(rgb).all() and torch.isfinite(means.grad).all()
    blender = subprocess.run(["blender", "--version"], check=True,
                             capture_output=True, text=True).stdout.splitlines()[0]
    print(json.dumps({"status": "passed", "python": platform.python_version(),
                      "gpu": torch.cuda.get_device_name(0), "torch": torch.__version__,
                      "cuda": torch.version.cuda, "gsplat": version("gsplat"),
                      "cuda_arch_list": os.environ.get("TORCH_CUDA_ARCH_LIST"),
                      "extension_cache": os.environ.get("TORCH_EXTENSIONS_DIR"),
                      "diffusers": version("diffusers"), "transformers": version("transformers"),
                      "blender": blender, "gsplat_forward_backward": True}, indent=2))


if __name__ == "__main__":
    main()
