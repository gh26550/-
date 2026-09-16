#!/usr/bin/env python3
"""
Step 6-prep: scene_graph.json から Unity配置用 layout_for_unity.json を作る。

入力:
  outputs/scene_graphs/<scene_id>/scene_graph.json

出力:
  outputs/scene_layouts/<scene_id>/layout_for_unity.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict


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


def unity_asset_info(scene_id: str, obj: Dict[str, Any]) -> Dict[str, Any]:
    oid = obj.get("id", "")
    rep = obj.get("representation", "")
    category = obj.get("category", "")

    if rep == "gaussian":
        return {
            "unity_type": "gaussian_splat",
            "asset_path": f"outputs/gaussian_assets/{scene_id}/{oid}/{oid}.ply",
            "renderer_name": oid,
        }

    if category in PROCEDURAL_CATEGORIES:
        return {
            "unity_type": "procedural_mesh",
            "asset_path": None,
            "primitive_category": category,
        }

    return {
        "unity_type": "mesh_gameobject",
        "asset_path": f"outputs/mesh_assets/{scene_id}/{oid}/{oid}.glb",
        "prefab_name": oid,
    }


def build_layout(scene_graph: Dict[str, Any]) -> Dict[str, Any]:
    scene_id = scene_graph.get("scene_id", "unknown_scene")
    unity_objects = []

    for obj in scene_graph.get("objects", []):
        unity_objects.append({
            "id": obj.get("id", ""),
            "name": obj.get("name", ""),
            "category": obj.get("category", ""),
            "representation": obj.get("representation", ""),
            "interaction_required": bool(obj.get("interaction_required", False)),
            "background_only": bool(obj.get("background_only", False)),
            "transform": obj.get("transform", {}),
            "asset": unity_asset_info(scene_id, obj),
        })

    return {
        "scene_id": scene_id,
        "scene_name": scene_graph.get("scene_name", ""),
        "room": scene_graph.get("room", {}),
        "coordinate_system": scene_graph.get("room", {}).get("coordinate_system", "Unity-like"),
        "hybrid_enabled": False,
        "unity_objects": unity_objects,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Assemble Unity layout JSON from scene_graph.json.")
    parser.add_argument("--scene_graph", type=str, required=True, help="Path to scene_graph.json")
    parser.add_argument("--output", type=str, required=True, help="Path to layout_for_unity.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_graph = load_json(Path(args.scene_graph))
    layout = build_layout(scene_graph)
    save_json(Path(args.output), layout)

    mesh_count = sum(1 for o in layout["unity_objects"] if o["representation"] == "mesh")
    gaussian_count = sum(1 for o in layout["unity_objects"] if o["representation"] == "gaussian")

    print(f"Saved Unity layout: {args.output}")
    print(f"mesh objects: {mesh_count}")
    print(f"gaussian objects: {gaussian_count}")


if __name__ == "__main__":
    main()
