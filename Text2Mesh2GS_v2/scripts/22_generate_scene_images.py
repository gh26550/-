#!/usr/bin/env python3
"""
Step 1: テキストから空間全体のコンセプト画像を生成する。

入力:
  data/scene_prompts.csv
  configs/scene_pipeline.yaml

出力例:
  outputs/scene_images/room_001/
    candidate_00.png
    candidate_01.png
    candidate_02.png
    candidate_03.png
    generation_meta.json

実行例:
  python scripts/22_generate_scene_images.py --config configs/scene_pipeline.yaml
  python scripts/22_generate_scene_images.py --config configs/scene_pipeline.yaml --scene_id room_001
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


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def clean_csv_value(value: Optional[str]) -> str:
    """
    CSVセルの値を整理する。
    - Noneを空文字にする
    - 前後の空白・改行を削除する
    - 余分な外側の二重引用符を削除する
    """
    if value is None:
        return ""

    s = value.strip()

    # 例: '"明るい部屋。"' のように、値そのものに引用符が残った場合に外す
    while len(s) >= 2 and s.startswith('"') and s.endswith('"'):
        s = s[1:-1].strip()

    return s


def read_scene_prompts(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Prompt CSV not found: {csv_path}")

    # ExcelやWindowsで作成されたCSVはCP932になることがあるため、
    # UTF-8だけでなくCP932 / Shift_JISも試す。
    encodings = ["utf-8-sig", "utf-8", "cp932", "shift_jis"]
    last_error: Optional[UnicodeDecodeError] = None

    for enc in encodings:
        try:
            rows: List[Dict[str, str]] = []

            with csv_path.open("r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)

                required = {"scene_id", "scene_name", "prompt"}
                missing = required - set(reader.fieldnames or [])
                if missing:
                    raise ValueError(f"Prompt CSV is missing columns: {sorted(missing)}")

                for row in reader:
                    scene_id = clean_csv_value(row.get("scene_id"))
                    scene_name = clean_csv_value(row.get("scene_name"))
                    prompt = clean_csv_value(row.get("prompt"))

                    if not scene_id or not prompt:
                        continue

                    rows.append({
                        "scene_id": scene_id,
                        "scene_name": scene_name,
                        "prompt": prompt,
                    })

            if not rows:
                raise ValueError(f"No valid scene prompts found in: {csv_path}")

            print(f"Loaded prompt CSV with encoding: {enc}")
            return rows

        except UnicodeDecodeError as e:
            last_error = e
            continue

    raise UnicodeDecodeError(
        "utf-8/cp932/shift_jis",
        b"",
        0,
        1,
        f"Could not decode {csv_path}. Last error: {last_error}"
    )


def resolve_project_path(config_path: Path, maybe_relative_path: str) -> Path:
    """configファイルの場所ではなく、実行中のプロジェクトルート基準で相対パスを解決する。"""
    p = Path(maybe_relative_path)
    if p.is_absolute():
        return p
    return Path.cwd() / p


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
        # accelerate が入っている環境ではVRAM使用量を抑えられる。
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(device)

    # xformersが入っていれば有効化する。入っていない場合は無視する。
    if device == "cuda":
        try:
            pipe.enable_xformers_memory_efficient_attention()
        except Exception:
            pass

    return pipe


def iter_target_scenes(rows: Iterable[Dict[str, str]], scene_id: Optional[str]) -> Iterable[Dict[str, str]]:
    for row in rows:
        if scene_id is None or row["scene_id"] == scene_id:
            yield row


def generate_scene_images(config: Dict[str, Any], config_path: Path, scene_id: Optional[str]) -> None:
    scene_input = config["scene_input"]
    gen_cfg = config["scene_image_generation"]
    project_cfg = config["project"]

    if not gen_cfg.get("enabled", True):
        print("scene_image_generation.enabled is false. Nothing to do.")
        return

    prompt_csv = resolve_project_path(config_path, scene_input["prompt_csv"])
    output_root = resolve_project_path(config_path, project_cfg.get("output_root", "outputs"))
    out_base = output_root / "scene_images"
    out_base.mkdir(parents=True, exist_ok=True)

    rows = read_scene_prompts(prompt_csv)
    targets = list(iter_target_scenes(rows, scene_id))
    if not targets:
        raise ValueError(f"No scene matched scene_id={scene_id!r}")

    model_id = gen_cfg["model_id"]
    width = int(gen_cfg.get("width", 1024))
    height = int(gen_cfg.get("height", 1024))
    num_images = int(gen_cfg.get("num_images", 4))
    seed = int(gen_cfg.get("seed", 0))
    device = choose_device(str(gen_cfg.get("device", "auto")))
    low_vram = bool(gen_cfg.get("low_vram", True))
    num_inference_steps = int(gen_cfg.get("num_inference_steps", 40))
    guidance_scale = float(gen_cfg.get("guidance_scale", 6.5))
    prompt_template = str(gen_cfg.get("prompt_template", "{prompt}"))
    negative_prompt = str(gen_cfg.get("negative_prompt", ""))

    print(f"Loading SDXL pipeline: {model_id}")
    print(f"Device: {device}, low_vram={low_vram}")
    pipe = build_pipeline(model_id=model_id, device=device, low_vram=low_vram)

    for scene in targets:
        sid = scene["scene_id"]
        scene_dir = out_base / sid
        scene_dir.mkdir(parents=True, exist_ok=True)

        full_prompt = prompt_template.format(prompt=scene["prompt"], scene_name=scene.get("scene_name", ""))
        print(f"\nGenerating scene images for {sid}: {scene.get('scene_name', '')}")
        print(f"Prompt: {full_prompt}")

        saved_files: List[str] = []
        for i in range(num_images):
            image_seed = seed + i
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
            out_path = scene_dir / f"candidate_{i:02d}.png"
            image.save(out_path)
            saved_files.append(str(out_path.as_posix()))
            print(f"Saved: {out_path}")

        meta = {
            "scene_id": sid,
            "scene_name": scene.get("scene_name", ""),
            "source_prompt": scene["prompt"],
            "generated_prompt": full_prompt,
            "negative_prompt": negative_prompt,
            "model_id": model_id,
            "width": width,
            "height": height,
            "num_images": num_images,
            "seed_start": seed,
            "num_inference_steps": num_inference_steps,
            "guidance_scale": guidance_scale,
            "files": saved_files,
        }
        with (scene_dir / "generation_meta.json").open("w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate room-level concept images from scene prompts.")
    parser.add_argument("--config", type=str, required=True, help="Path to configs/scene_pipeline.yaml")
    parser.add_argument("--scene_id", type=str, default=None, help="Optional scene_id to generate only one scene")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)
    generate_scene_images(config=config, config_path=config_path, scene_id=args.scene_id)


if __name__ == "__main__":
    main()
