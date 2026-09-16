#!/usr/bin/env python3
"""
Step 4-2: 生成済みの単体物体画像から、Trellis に渡すジョブ一覧を作る。

このスクリプトは Trellis 本体を実行しない。
Trellis へ渡す input_image と expected_glb の対応表を作る。

実行例:
  python scripts/29_prepare_trellis_jobs.py \
    --mesh_csv outputs/scene_jobs/room_001/mesh_object_prompts.csv \
    --object_image_root outputs/object_images \
    --output_json outputs/scene_jobs/room_001/trellis_jobs.json \
    --output_csv outputs/scene_jobs/room_001/trellis_jobs.csv
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, List, Optional


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
                rows = [{k: clean_csv_value(v) for k, v in row.items()} for row in reader]
            print(f"Loaded mesh prompt CSV with encoding: {enc}")
            return rows
        except UnicodeDecodeError as e:
            last_error = e
            continue
    raise UnicodeDecodeError("utf-8/cp932/shift_jis", b"", 0, 1, f"Could not decode {csv_path}: {last_error}")


def find_input_image(object_dir: Path, preferred_index: int) -> Optional[Path]:
    # Trellis には背景透過PNGがある場合はそちらを優先する。
    candidates = [
        object_dir / f"candidate_{preferred_index:02d}_rgba.png",
        object_dir / f"candidate_{preferred_index:02d}.png",
        object_dir / "candidate_00_rgba.png",
        object_dir / "candidate_00.png",
    ]
    for p in candidates:
        if p.exists():
            return p
    return None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Prepare Trellis job list from generated object images.")
    parser.add_argument("--mesh_csv", type=str, required=True, help="Path to mesh_object_prompts.csv")
    parser.add_argument("--object_image_root", type=str, default="outputs/object_images", help="Root directory of generated object images")
    parser.add_argument("--output_json", type=str, required=True, help="Output trellis_jobs.json")
    parser.add_argument("--output_csv", type=str, required=True, help="Output trellis_jobs.csv")
    parser.add_argument("--candidate_index", type=int, default=0, help="Preferred candidate image index")
    parser.add_argument("--resolution", type=int, default=512, help="Recommended Trellis resolution")
    parser.add_argument("--decimation_target", type=int, default=500000, help="Recommended mesh decimation target")
    parser.add_argument("--texture_size", type=int, default=2048, help="Recommended texture size")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    mesh_csv = Path(args.mesh_csv)
    object_image_root = Path(args.object_image_root)
    output_json = Path(args.output_json)
    output_csv = Path(args.output_csv)

    rows = read_csv_flexible(mesh_csv)
    jobs = []

    for row in rows:
        scene_id = row.get("scene_id", "unknown_scene")
        object_id = row.get("object_id") or row.get("id")
        if not object_id:
            continue

        object_dir = object_image_root / scene_id / object_id
        input_image = find_input_image(object_dir, args.candidate_index)
        expected_glb = row.get("expected_glb") or f"outputs/mesh_assets/{scene_id}/{object_id}/{object_id}.glb"
        output_dir = row.get("output_dir") or str(Path(expected_glb).parent)

        job = {
            "scene_id": scene_id,
            "object_id": object_id,
            "name": row.get("name", object_id),
            "category": row.get("category", ""),
            "input_image": str(input_image.as_posix()) if input_image else None,
            "output_dir": output_dir,
            "expected_glb": expected_glb,
            "trellis_settings": {
                "resolution": args.resolution,
                "decimation_target": args.decimation_target,
                "texture_size": args.texture_size,
            },
            "status": "ready" if input_image else "missing_input_image",
        }
        jobs.append(job)

    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_csv.parent.mkdir(parents=True, exist_ok=True)

    with output_json.open("w", encoding="utf-8") as f:
        json.dump({"job_type": "trellis_mesh_generation", "num_jobs": len(jobs), "jobs": jobs}, f, ensure_ascii=False, indent=2)

    fieldnames = ["scene_id", "object_id", "name", "category", "input_image", "output_dir", "expected_glb", "resolution", "decimation_target", "texture_size", "status"]
    with output_csv.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for job in jobs:
            settings = job["trellis_settings"]
            writer.writerow({
                "scene_id": job["scene_id"],
                "object_id": job["object_id"],
                "name": job["name"],
                "category": job["category"],
                "input_image": job["input_image"] or "",
                "output_dir": job["output_dir"],
                "expected_glb": job["expected_glb"],
                "resolution": settings["resolution"],
                "decimation_target": settings["decimation_target"],
                "texture_size": settings["texture_size"],
                "status": job["status"],
            })

    ready = sum(1 for j in jobs if j["status"] == "ready")
    missing = len(jobs) - ready
    print(f"Saved: {output_json}")
    print(f"Saved: {output_csv}")
    print(f"Ready jobs: {ready}")
    print(f"Missing input image jobs: {missing}")


if __name__ == "__main__":
    main()
