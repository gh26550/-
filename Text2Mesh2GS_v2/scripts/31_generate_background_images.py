#!/usr/bin/env python3
"""Generate background candidates from step 26; model inference only with --execute."""
import argparse
import json
from pathlib import Path


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--prompts", required=True)
    p.add_argument("--output-root", default="outputs/background_images")
    p.add_argument("--model", default="stabilityai/stable-diffusion-xl-base-1.0")
    p.add_argument("--seed", type=int, default=12345)
    p.add_argument("--num-images", type=int, default=2)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    data = json.loads(Path(args.prompts).read_text(encoding="utf-8-sig"))
    if args.num_images < 1:
        raise ValueError("num-images must be positive")
    if not args.execute:
        print(json.dumps({"status": "planned", "scene_id": data["scene_id"], "objects": [x["object_id"] for x in data["prompts"]]}, ensure_ascii=False))
        return
    import torch
    from diffusers import StableDiffusionXLPipeline
    if not torch.cuda.is_available():
        raise RuntimeError("This image-generation entry point requires CUDA")
    pipe = StableDiffusionXLPipeline.from_pretrained(args.model, torch_dtype=torch.float16, use_safetensors=True).to("cuda")
    pipe.enable_attention_slicing()
    for row in data["prompts"]:
        folder = Path(args.output_root) / data["scene_id"] / row["object_id"]
        folder.mkdir(parents=True, exist_ok=True)
        for i in range(args.num_images):
            image = pipe(prompt=row["background_prompt"], negative_prompt=data.get("negative_prompt", ""),
                         width=1024, height=768, num_inference_steps=30, guidance_scale=6.5,
                         generator=torch.Generator(device="cuda").manual_seed(args.seed+i)).images[0]
            image.save(folder / f"candidate_{i:02d}.png")
        (folder / "generation_meta.json").write_text(json.dumps({"prompt": row["background_prompt"], "model": args.model, "seed": args.seed}, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
