#!/usr/bin/env python3
"""
Generate an initial spatial_constraints.json from a scene_graph.json.

This is a rule-based starter version inspired by HOLODECK-style spatial
constraint generation. Later, the same JSON schema can be produced by an LLM/VLM.

Example:
  python scripts/34_generate_spatial_constraints.py \
    --scene_graph outputs/scene_graphs/room_001/scene_graph.json \
    --output data/spatial_constraints_room_001.json
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional


def load_json(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def save_json(data: Dict[str, Any], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)
    print(f"[SAVED] {path}")


def find_first(objects: Iterable[Dict[str, Any]], categories: Iterable[str]) -> Optional[Dict[str, Any]]:
    cats = set(categories)
    for obj in objects:
        if obj.get("category") in cats:
            return obj
    return None


def exists(objects_by_id: Dict[str, Dict[str, Any]], object_id: Optional[str]) -> bool:
    return bool(object_id) and object_id in objects_by_id


def add_constraint(
    constraints: List[Dict[str, Any]],
    objects_by_id: Dict[str, Dict[str, Any]],
    type_: str,
    relation: str,
    source: Optional[str],
    target: Optional[str],
    weight: float = 1.0,
    note: str = "",
) -> None:
    if not exists(objects_by_id, source) or not exists(objects_by_id, target):
        return
    if source == target:
        return
    item = {
        "type": type_,
        "relation": relation,
        "source": source,
        "target": target,
        "weight": weight,
    }
    if note:
        item["note"] = note
    if item not in constraints:
        constraints.append(item)


def generate_constraints(scene_graph: Dict[str, Any]) -> Dict[str, Any]:
    objects = scene_graph.get("objects", [])
    objects_by_id = {obj.get("id"): obj for obj in objects if obj.get("id")}

    table = find_first(objects, ["table", "coffee_table"])
    sofa = find_first(objects, ["sofa"])
    chair = find_first(objects, ["chair"])
    vase = find_first(objects, ["vase"])
    plant = find_first(objects, ["plant"])
    bookshelf = find_first(objects, ["bookshelf", "shelf"])
    window = find_first(objects, ["window_frame", "window"])
    outside = find_first(objects, ["outside_view", "sky", "distant_building", "distant_tree"])

    wall_left = objects_by_id.get("wall_left_01")
    wall_back = objects_by_id.get("wall_back_01")

    constraints: List[Dict[str, Any]] = []

    if table and vase:
        add_constraint(
            constraints,
            objects_by_id,
            "vertical",
            "on",
            vase["id"],
            table["id"],
            1.0,
            "Small decorative object should rest on the table surface.",
        )

    if table and sofa:
        add_constraint(constraints, objects_by_id, "rotation", "face to", sofa["id"], table["id"], 1.0)
        add_constraint(constraints, objects_by_id, "distance", "near", sofa["id"], table["id"], 0.7)

    if table and chair:
        add_constraint(constraints, objects_by_id, "rotation", "face to", chair["id"], table["id"], 1.0)
        add_constraint(constraints, objects_by_id, "distance", "near", chair["id"], table["id"], 0.7)

    if bookshelf:
        target_wall = wall_left or wall_back
        if target_wall:
            add_constraint(
                constraints,
                objects_by_id,
                "relative",
                "against",
                bookshelf["id"],
                target_wall["id"],
                1.0,
                "Large storage furniture should be placed along a wall.",
            )

    if plant and window:
        add_constraint(
            constraints,
            objects_by_id,
            "distance",
            "near",
            plant["id"],
            window["id"],
            1.0,
            "Indoor plant should be placed near daylight/window area.",
        )

    if outside and window:
        add_constraint(
            constraints,
            objects_by_id,
            "relative",
            "behind",
            outside["id"],
            window["id"],
            1.0,
            "Gaussian outdoor background should be behind the physical window frame.",
        )

    return {
        "scene_id": scene_graph.get("scene_id", ""),
        "description": "Initial rule-based spatial constraints generated from scene_graph categories.",
        "coordinate_system": scene_graph.get("room", {}).get(
            "coordinate_system", "Unity-like: x=right, y=up, z=forward/depth"
        ),
        "constraints": constraints,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scene_graph", required=True, help="Path to scene_graph.json")
    parser.add_argument("--output", required=True, help="Path to output spatial_constraints.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    scene_graph = load_json(Path(args.scene_graph))
    constraints = generate_constraints(scene_graph)
    save_json(constraints, Path(args.output))


if __name__ == "__main__":
    main()
