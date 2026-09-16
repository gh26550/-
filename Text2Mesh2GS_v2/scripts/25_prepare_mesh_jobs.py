#!/usr/bin/env python3
"""
Step 4-prep: mesh_jobs.json を、既存の単体Mesh生成パイプラインに流せるCSVへ変換する。

入力:
  outputs/scene_jobs/<scene_id>/mesh_jobs.json

出力:
  outputs/scene_jobs/<scene_id>/mesh_object_prompts.csv
  outputs/scene_jobs/<scene_id>/procedural_mesh_jobs.json

方針:
  - sofa/chair/table/bookshelf/vase/plant などは Trellis 用の単体物体プロンプトへ送る
  - floor/wall/ceiling/window_frame/rug などは Unity 側でプリミティブ生成する
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Tuple


PROCEDURAL_CATEGORIES = {
    "floor",
    "wall",
    "ceiling",
    "window_frame",
    "rug",
}


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(path: Path, data: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def split_mesh_jobs(mesh_jobs: Dict[str, Any]) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    trellis_jobs: List[Dict[str, Any]] = []
    procedural_jobs: List[Dict[str, Any]] = []

    for job in mesh_jobs.get("jobs", []):
        category = job.get("category", "")
        if category in PROCEDURAL_CATEGORIES:
            procedural_jobs.append(job)
        else:
            trellis_jobs.append(job)

    return trellis_jobs, procedural_jobs


def write_mesh_prompt_csv(path: Path, scene_id: str, jobs: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)

    fieldnames = [
        "scene_id",
        "object_id",
        "name",
        "category",
        "object_prompt",
        "image_prompt",
        "output_dir",
        "expected_glb",
        "interaction_required",
        "pos_x",
        "pos_y",
        "pos_z",
        "rot_x",
        "rot_y",
        "rot_z",
        "scale_x",
        "scale_y",
        "scale_z",
    ]

    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()

        for job in jobs:
            transform = job.get("transform", {})
            pos = transform.get("position", [0.0, 0.0, 0.0])
            rot = transform.get("rotation_euler", [0.0, 0.0, 0.0])
            scale = transform.get("scale", [1.0, 1.0, 1.0])
            expected = job.get("expected_outputs", {})

            object_prompt = job.get("object_prompt", job.get("name", ""))
            image_prompt = (
                f"{object_prompt}, single object, centered, front view, "
                "isolated object, white background, no text, no watermark"
            )

            writer.writerow({
                "scene_id": scene_id,
                "object_id": job.get("object_id", ""),
                "name": job.get("name", ""),
                "category": job.get("category", ""),
                "object_prompt": object_prompt,
                "image_prompt": image_prompt,
                "output_dir": f"outputs/mesh_assets/{scene_id}/{job.get('object_id', '')}",
                "expected_glb": expected.get("glb", ""),
                "interaction_required": str(bool(job.get("interaction_required", False))).lower(),
                "pos_x": pos[0],
                "pos_y": pos[1],
                "pos_z": pos[2],
                "rot_x": rot[0],
                "rot_y": rot[1],
                "rot_z": rot[2],
                "scale_x": scale[0],
                "scale_y": scale[1],
                "scale_z": scale[2],
            })


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare mesh object prompts for the single-object pipeline.")
    parser.add_argument("--jobs", type=str, required=True, help="Path to mesh_jobs.json")
    parser.add_argument("--output_csv", type=str, required=True, help="Path to mesh_object_prompts.csv")
    parser.add_argument(
        "--procedural_output",
        type=str,
        default=None,
        help="Optional path to procedural_mesh_jobs.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    jobs_path = Path(args.jobs)
    output_csv = Path(args.output_csv)
    procedural_output = Path(args.procedural_output) if args.procedural_output else output_csv.parent / "procedural_mesh_jobs.json"

    mesh_jobs = load_json(jobs_path)
    scene_id = mesh_jobs.get("scene_id", "unknown_scene")

    trellis_jobs, procedural_jobs = split_mesh_jobs(mesh_jobs)

    write_mesh_prompt_csv(output_csv, scene_id, trellis_jobs)

    save_json(procedural_output, {
        "scene_id": scene_id,
        "scene_name": mesh_jobs.get("scene_name", ""),
        "job_type": "procedural_mesh_generation",
        "num_jobs": len(procedural_jobs),
        "jobs": procedural_jobs,
    })

    print(f"Saved Trellis mesh prompt CSV: {output_csv}")
    print(f"Saved procedural mesh jobs: {procedural_output}")
    print(f"Trellis/object mesh jobs: {len(trellis_jobs)}")
    print(f"Procedural mesh jobs: {len(procedural_jobs)}")


if __name__ == "__main__":
    main()
