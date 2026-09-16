#!/usr/bin/env python3
"""
Step 3: scene_graph.json を Meshジョブと Gaussianジョブに分割する。

入力:
  outputs/scene_graphs/<scene_id>/scene_graph.json

出力:
  outputs/scene_jobs/<scene_id>/mesh_jobs.json
  outputs/scene_jobs/<scene_id>/gaussian_jobs.json

実行例:
  python scripts/24_split_scene_jobs.py \
    --scene_graph outputs/scene_graphs/room_001/scene_graph.json \
    --output_dir outputs/scene_jobs/room_001
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


def make_mesh_job(scene_graph: Dict[str, Any], objects: List[Dict[str, Any]]) -> Dict[str, Any]:
    jobs = []
    for o in objects:
        jobs.append({
            "object_id": o["id"],
            "name": o["name"],
            "category": o["category"],
            "interaction_required": o.get("interaction_required", False),
            "background_only": o.get("background_only", False),
            "transform": o["transform"],
            "object_prompt": o.get("asset_prompt", o["name"]),
            "pipeline": {
                "type": "mesh",
                "steps": ["object text", "front image", "Trellis", "GLB mesh", "clean", "Unity conversion"],
            },
            "expected_outputs": {
                "glb": f"outputs/mesh_assets/{scene_graph['scene_id']}/{o['id']}/{o['id']}.glb",
                "unity_prefab_name": o["id"],
            },
        })

    return {
        "scene_id": scene_graph["scene_id"],
        "scene_name": scene_graph.get("scene_name", ""),
        "job_type": "mesh_generation",
        "hybrid_enabled": False,
        "num_jobs": len(jobs),
        "jobs": jobs,
    }


def make_gaussian_job(scene_graph: Dict[str, Any], objects: List[Dict[str, Any]]) -> Dict[str, Any]:
    background_parts = []
    for o in objects:
        background_parts.append(o.get("asset_prompt", o.get("name", "")))

    background_prompt = "; ".join([p for p in background_parts if p])
    if scene_graph.get("source_prompt"):
        background_prompt = f"{background_prompt}; based on scene: {scene_graph['source_prompt']}"

    jobs = []
    for o in objects:
        jobs.append({
            "object_id": o["id"],
            "name": o["name"],
            "category": o["category"],
            "interaction_required": o.get("interaction_required", False),
            "background_only": o.get("background_only", True),
            "transform": o["transform"],
            "background_prompt": o.get("asset_prompt", o["name"]),
            "pipeline": {
                "type": "gaussian",
                "steps": ["background text", "background image/simple scene", "multi-view rendering", "Gaussian training", "PLY export"],
            },
            "expected_outputs": {
                "ply": f"outputs/gaussian_assets/{scene_graph['scene_id']}/{o['id']}/{o['id']}.ply",
                "unity_renderer_name": o["id"],
            },
        })

    return {
        "scene_id": scene_graph["scene_id"],
        "scene_name": scene_graph.get("scene_name", ""),
        "job_type": "gaussian_background_generation",
        "hybrid_enabled": False,
        "combined_background_prompt": background_prompt,
        "num_jobs": len(jobs),
        "jobs": jobs,
    }


def split_scene_jobs(scene_graph: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    objects = scene_graph.get("objects", [])
    mesh_objects = [o for o in objects if o.get("representation") == "mesh"]
    gaussian_objects = [o for o in objects if o.get("representation") == "gaussian"]

    invalid = [o for o in objects if o.get("representation") not in {"mesh", "gaussian"}]
    if invalid:
        ids = [o.get("id", "<unknown>") for o in invalid]
        raise ValueError(f"Invalid representation found. Hybrid is disabled. Invalid object ids: {ids}")

    return {
        "mesh_jobs": make_mesh_job(scene_graph, mesh_objects),
        "gaussian_jobs": make_gaussian_job(scene_graph, gaussian_objects),
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Split scene_graph.json into mesh_jobs.json and gaussian_jobs.json.")
    parser.add_argument("--scene_graph", type=str, required=True, help="Path to scene_graph.json")
    parser.add_argument("--output_dir", type=str, required=True, help="Directory to save mesh_jobs.json and gaussian_jobs.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_graph_path = Path(args.scene_graph)
    output_dir = Path(args.output_dir)

    scene_graph = load_json(scene_graph_path)
    split = split_scene_jobs(scene_graph)

    mesh_path = output_dir / "mesh_jobs.json"
    gaussian_path = output_dir / "gaussian_jobs.json"

    save_json(mesh_path, split["mesh_jobs"])
    save_json(gaussian_path, split["gaussian_jobs"])

    print(f"Saved mesh jobs: {mesh_path}")
    print(f"Saved gaussian jobs: {gaussian_path}")
    print(f"mesh jobs: {split['mesh_jobs']['num_jobs']}")
    print(f"gaussian jobs: {split['gaussian_jobs']['num_jobs']}")


if __name__ == "__main__":
    main()
