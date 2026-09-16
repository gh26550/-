#!/usr/bin/env python3
"""
Step 5-prep: gaussian_jobs.json から背景生成用プロンプトを作る。

入力:
  outputs/scene_jobs/<scene_id>/gaussian_jobs.json

出力:
  outputs/scene_jobs/<scene_id>/gaussian_background_prompts.json
  outputs/scene_jobs/<scene_id>/background_prompt.txt
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def build_prompt(job: Dict[str, Any]) -> str:
    base = job.get("background_prompt", job.get("name", ""))
    category = job.get("category", "background")

    # Gaussian背景は「近距離物体」ではなく、窓外・遠景・雰囲気として固定する。
    return (
        f"{base}, {category}, distant background only, seen through a large window, "
        "soft daylight, no foreground furniture, no indoor objects, no people, "
        "wide background plate, coherent outdoor depth"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Gaussian background generation prompts.")
    parser.add_argument("--jobs", type=str, required=True, help="Path to gaussian_jobs.json")
    parser.add_argument("--output_dir", type=str, required=True, help="Output directory")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs_path = Path(args.jobs)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    gaussian_jobs = load_json(jobs_path)
    scene_id = gaussian_jobs.get("scene_id", "unknown_scene")

    prompts: List[Dict[str, Any]] = []
    for job in gaussian_jobs.get("jobs", []):
        prompts.append({
            "scene_id": scene_id,
            "object_id": job.get("object_id", ""),
            "name": job.get("name", ""),
            "category": job.get("category", ""),
            "background_prompt": build_prompt(job),
            "transform": job.get("transform", {}),
            "expected_ply": job.get("expected_outputs", {}).get("ply", ""),
        })

    combined_prompt = gaussian_jobs.get("combined_background_prompt", "")
    if prompts:
        combined_prompt = prompts[0]["background_prompt"]

    output_json = {
        "scene_id": scene_id,
        "scene_name": gaussian_jobs.get("scene_name", ""),
        "job_type": "gaussian_background_prompt_generation",
        "num_prompts": len(prompts),
        "combined_background_prompt": combined_prompt,
        "negative_prompt": (
            "indoor furniture, sofa, chair, table, bookshelf, vase, potted plant, "
            "close-up, macro, single object, people, text, watermark, distorted buildings"
        ),
        "prompts": prompts,
    }

    json_path = output_dir / "gaussian_background_prompts.json"
    txt_path = output_dir / "background_prompt.txt"

    save_json(json_path, output_json)
    txt_path.write_text(combined_prompt + "\n", encoding="utf-8")

    print(f"Saved Gaussian background prompts: {json_path}")
    print(f"Saved background prompt text: {txt_path}")
    print(f"Gaussian prompt jobs: {len(prompts)}")


if __name__ == "__main__":
    main()
