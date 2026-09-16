#!/usr/bin/env python3
"""
Step 4-1: mesh_object_prompts.csv から、Trellis に入力する単体物体の正面画像を生成する。

入力例:
  outputs/scene_jobs/room_001/mesh_object_prompts.csv

出力例:
  outputs/object_images/room_001/chair_01/
    candidate_00.png
    candidate_00_rgba.png  # --remove_bg を使った場合
    generation_meta.json

実行例:
  python scripts/28_generate_object_front_images.py \
    --config configs/scene_pipeline.yaml \
    --mesh_csv outputs/scene_jobs/room_001/mesh_object_prompts.csv

  python scripts/28_generate_object_front_images.py \
    --config configs/scene_pipeline.yaml \
    --mesh_csv outputs/scene_jobs/room_001/mesh_object_prompts.csv \
    --remove_bg
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

import torch
import yaml
from diffusers import StableDiffusionXLPipeline
from PIL import Image


def load_yaml(path: Optional[Path]) -> Dict[str, Any]:
    if path is None:
        return {}
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def resolve_project_path(maybe_relative_path: str) -> Path:
    p = Path(maybe_relative_path)
    if p.is_absolute():
        return p
    return Path.cwd() / p


def clean_csv_value(value: Optional[str]) -> str:
    if value is None:
        return ""
    s = value.strip()
    while len(s) >= 2 and s.startswith('"') and s.endswith('"'):
        s = s[1:-1].strip()
    return s


def read_csv_flexible(csv_path: Path) -> List[Dict[str, str]]:
    encodings = ["utf-8-sig", "utf-8", "cp932", "shift_jis"]
    last_error: Optional[UnicodeDecodeError] = None

    for enc in encodings:
        try:
            with csv_path.open("r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                if not reader.fieldnames:
                    raise ValueError(f"CSV has no header: {csv_path}")
                rows: List[Dict[str, str]] = []
                for row in reader:
                    rows.append({k: clean_csv_value(v) for k, v in row.items()})
            print(f"Loaded mesh prompt CSV with encoding: {enc}")
            return rows
        except UnicodeDecodeError as e:
            last_error = e
            continue

    raise UnicodeDecodeError(
        "utf-8/cp932/shift_jis",
        b"",
        0,
        1,
        f"Could not decode {csv_path}. Last error: {last_error}",
    )


def choose_device(device_setting: str) -> str:
    if device_setting != "auto":
        return device_setting
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def build_pipeline(model_id: str, device: str, low_vram: bool) -> StableDiffusionXLPipeline:
    if device == "cuda":
        dtype = torch.float16
        torch.backends.cuda.matmul.allow_tf32 = True
    elif device == "mps":
        dtype = torch.float16
    else:
        dtype = torch.float32

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=dtype,
        use_safetensors=True,
    )

    if device == "cuda" and low_vram:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    return pipe


def remove_background_if_available(image: Image.Image) -> Image.Image:
    """rembg が入っている場合だけ背景除去する。未導入なら例外を出す。"""
    try:
        from rembg import remove
    except Exception as e:
        raise RuntimeError(
            "rembg is not installed. Install it with: pip install rembg onnxruntime"
        ) from e

    rgba = remove(image.convert("RGBA"))
    if not isinstance(rgba, Image.Image):
        rgba = Image.open(rgba).convert("RGBA")
    return rgba


def get_object_id(row: Dict[str, str]) -> str:
    return row.get("object_id") or row.get("id") or row.get("objectId") or "unknown_object"


def get_prompt(row: Dict[str, str]) -> str:
    return (
        row.get("image_prompt")
        or row.get("object_prompt")
        or row.get("asset_prompt")
        or row.get("name")
        or "single object"
    )


def generate_images(
    rows: List[Dict[str, str]],
    config: Dict[str, Any],
    output_root: Path,
    num_images_override: Optional[int],
    remove_bg: bool,
    only_object_id: Optional[str],
) -> None:
    gen_cfg = config.get("object_image_generation", {})
    scene_gen_cfg = config.get("scene_image_generation", {})

    model_id = gen_cfg.get("model_id") or scene_gen_cfg.get("model_id") or "stabilityai/stable-diffusion-xl-base-1.0"
    width = int(gen_cfg.get("width", 768))
    height = int(gen_cfg.get("height", 768))
    num_images = int(num_images_override or gen_cfg.get("num_images", 2))
    seed = int(gen_cfg.get("seed", 1000))
    device = choose_device(str(gen_cfg.get("device", scene_gen_cfg.get("device", "auto"))))
    low_vram = bool(gen_cfg.get("low_vram", scene_gen_cfg.get("low_vram", True)))
    num_inference_steps = int(gen_cfg.get("num_inference_steps", 40))
    guidance_scale = float(gen_cfg.get("guidance_scale", 7.0))

    prompt_template = str(gen_cfg.get(
        "prompt_template",
        "3D asset reference image of {prompt}, orthographic front view, full object visible, centered, single object, isolated on pure white background, clean silhouette, realistic material, no environment"
    ))
    negative_prompt = str(gen_cfg.get(
        "negative_prompt",
        "room, indoor scene, furniture set, multiple objects, collection, close-up, cropped, cut off, partial object, hands, people, text, watermark, logo, blurry, low quality, black background, transparent-looking background"
    ))

    targets = []
    for row in rows:
        oid = get_object_id(row)
        if only_object_id is None or oid == only_object_id:
            targets.append(row)
    if not targets:
        raise ValueError(f"No target object matched only_object_id={only_object_id!r}")

    print(f"Loading SDXL pipeline: {model_id}")
    print(f"Device: {device}, low_vram={low_vram}")
    pipe = build_pipeline(model_id=model_id, device=device, low_vram=low_vram)

    for obj_index, row in enumerate(targets):
        scene_id = row.get("scene_id", "unknown_scene")
        oid = get_object_id(row)
        name = row.get("name", oid)
        category = row.get("category", "")
        base_prompt = get_prompt(row)
        full_prompt = prompt_template.format(prompt=base_prompt, name=name, category=category)

        obj_dir = output_root / scene_id / oid
        obj_dir.mkdir(parents=True, exist_ok=True)

        print(f"\nGenerating object images for {oid}: {name}")
        print(f"Prompt: {full_prompt}")

        saved_files: List[str] = []
        for i in range(num_images):
            image_seed = seed + obj_index * 100 + i
            generator_device = device if device in {"cuda", "cpu"} else "cpu"
            generator = torch.Generator(device=generator_device).manual_seed(image_seed)

            result = pipe(
                prompt=full_prompt,
                negative_prompt=negative_prompt,
                width=width,
                height=height,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                generator=generator,
            )
            image = result.images[0]
            out_path = obj_dir / f"candidate_{i:02d}.png"
            image.save(out_path)
            saved_files.append(str(out_path.as_posix()))
            print(f"Saved: {out_path}")

            if remove_bg:
                try:
                    rgba = remove_background_if_available(image)
                    rgba_path = obj_dir / f"candidate_{i:02d}_rgba.png"
                    rgba.save(rgba_path)
                    saved_files.append(str(rgba_path.as_posix()))
                    print(f"Saved: {rgba_path}")
                except RuntimeError as e:
                    print(f"Background removal skipped: {e}")
                    remove_bg = False

        meta = {
            "scene_id": scene_id,
            "object_id": oid,
            "name": name,
            "category": category,
            "source_prompt": base_prompt,
            "generated_prompt": full_prompt,
            "negative_prompt": negative_prompt,
            "model_id": model_id,
            "width": width,
            "height": height,
            "num_images": num_images,
            "seed_base": seed,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "files": saved_files,
        }
        with (obj_dir / "generation_meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate front-view object images for Trellis input.")
    parser.add_argument("--config", type=str, default=None, help="Path to configs/scene_pipeline.yaml")
    parser.add_argument("--mesh_csv", type=str, required=True, help="Path to mesh_object_prompts.csv")
    parser.add_argument("--output_root", type=str, default="outputs/object_images", help="Output root directory")
    parser.add_argument("--num_images", type=int, default=None, help="Override number of images per object")
    parser.add_argument("--object_id", type=str, default=None, help="Generate only one object_id")
    parser.add_argument("--remove_bg", action="store_true", help="Also save transparent PNG using rembg")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_yaml(Path(args.config).resolve() if args.config else None)
    mesh_csv = resolve_project_path(args.mesh_csv)
    output_root = resolve_project_path(args.output_root)

    rows = read_csv_flexible(mesh_csv)
    generate_images(
        rows=rows,
        config=config,
        output_root=output_root,
        num_images_override=args.num_images,
        remove_bg=args.remove_bg,
        only_object_id=args.object_id,
    )


if __name__ == "__main__":
    main()
