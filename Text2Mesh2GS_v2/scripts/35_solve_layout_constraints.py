#!/usr/bin/env python3
"""
Produce a first automatic layout from layout_for_unity.json and spatial_constraints.json.

This Python solver is intentionally simple: it does not know the exact Unity
Renderer.bounds of generated GLBs. Use it to create a semantic initial layout.
Then use the Unity C# auto-placer for real bounds correction.

Example:
  python scripts/35_solve_layout_constraints.py \
    --layout outputs/scene_layouts/room_001/layout_for_unity.json \
    --constraints data/spatial_constraints_room_001.json \
    --output outputs/scene_layouts/room_001/layout_for_unity_auto.json \
    --report outputs/scene_layouts/room_001/placement_report.json
"""

from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, List, Tuple

Vec3 = Tuple[float, float, float]


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


def objects_by_id(layout: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    return {obj["id"]: obj for obj in layout.get("unity_objects", []) if obj.get("id")}


def set_pos(obj: Dict[str, Any], x: float, y: float, z: float) -> None:
    obj["transform"]["position"] = [round(x, 4), round(y, 4), round(z, 4)]


def set_rot_y(obj: Dict[str, Any], yaw_deg: float) -> None:
    rot = obj["transform"].setdefault("rotation_euler", [0.0, 0.0, 0.0])
    rot[0] = 0.0
    rot[1] = round(yaw_deg, 4)
    rot[2] = 0.0


def get_pos(obj: Dict[str, Any]) -> Vec3:
    p = obj.get("transform", {}).get("position", [0.0, 0.0, 0.0])
    return float(p[0]), float(p[1]), float(p[2])


def approx_size(obj: Dict[str, Any]) -> Vec3:
    """Approximate physical size in metres. Unity C# will later replace this with real bounds."""
    category = obj.get("category", "")
    defaults = {
        "sofa": (1.8, 0.8, 0.9),
        "chair": (0.55, 0.9, 0.55),
        "table": (1.2, 0.45, 1.2),
        "bookshelf": (1.2, 1.7, 0.4),
        "vase": (0.28, 0.45, 0.28),
        "plant": (0.7, 1.2, 0.7),
        "rug": (2.5, 0.03, 2.0),
        "window_frame": (3.2, 2.2, 0.1),
        "outside_view": (3.6, 2.4, 0.05),
    }
    if category in defaults:
        return defaults[category]
    scale = obj.get("transform", {}).get("scale", [1.0, 1.0, 1.0])
    return float(scale[0]), float(scale[1]), float(scale[2])


def yaw_face_to(source: Dict[str, Any], target: Dict[str, Any]) -> float:
    sx, _, sz = get_pos(source)
    tx, _, tz = get_pos(target)
    dx = tx - sx
    dz = tz - sz
    return math.degrees(math.atan2(dx, dz))


def xz_aabb(obj: Dict[str, Any]) -> Tuple[float, float, float, float]:
    x, _, z = get_pos(obj)
    sx, _, sz = approx_size(obj)
    return x - sx * 0.5, x + sx * 0.5, z - sz * 0.5, z + sz * 0.5


def intersects_xz(a: Dict[str, Any], b: Dict[str, Any], padding: float = 0.05) -> bool:
    aminx, amaxx, aminz, amaxz = xz_aabb(a)
    bminx, bmaxx, bminz, bmaxz = xz_aabb(b)
    return not (
        amaxx + padding < bminx
        or bmaxx + padding < aminx
        or amaxz + padding < bminz
        or bmaxz + padding < aminz
    )


def clamp_to_room(obj: Dict[str, Any], room_size: Vec3) -> None:
    width, _, depth = room_size
    sx, _, sz = approx_size(obj)
    x, y, z = get_pos(obj)
    x = max(-width * 0.5 + sx * 0.5, min(width * 0.5 - sx * 0.5, x))
    z = max(-depth * 0.5 + sz * 0.5, min(depth * 0.5 - sz * 0.5, z))
    set_pos(obj, x, y, z)


def solve(layout: Dict[str, Any], constraints: Dict[str, Any]) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    if layout.get("schema_version") == "2.0":
        raise ValueError("V2 uses Unity SemanticScenePlacerV2; Python 35 is only a legacy baseline")
    out = copy.deepcopy(layout)
    by_id = objects_by_id(out)
    room_size = tuple(out.get("room", {}).get("size", [6.0, 4.0, 6.0]))  # type: ignore
    room_w, _, room_d = room_size

    report: Dict[str, Any] = {"steps": [], "warnings": []}

    def has(obj_id: str) -> bool:
        return obj_id in by_id

    # Deterministic semantic initial layout for this room type.
    if has("table_01"):
        set_pos(by_id["table_01"], 0.0, 0.0, 0.4)
        report["steps"].append("Placed table_01 near the room center.")

    if has("rug_01"):
        set_pos(by_id["rug_01"], 0.2, 0.02, 0.5)

    if has("sofa_01"):
        set_pos(by_id["sofa_01"], 1.2, 0.0, 1.35)

    if has("chair_01"):
        set_pos(by_id["chair_01"], -1.35, 0.0, 0.25)

    if has("bookshelf_01"):
        sx, _, _ = approx_size(by_id["bookshelf_01"])
        set_pos(by_id["bookshelf_01"], -room_w * 0.5 + sx * 0.5 + 0.1, 0.0, 1.9)
        set_rot_y(by_id["bookshelf_01"], 90.0)

    if has("plant_01"):
        set_pos(by_id["plant_01"], room_w * 0.5 - 0.8, 0.0, room_d * 0.5 - 1.0)

    if has("window_frame_01"):
        set_pos(by_id["window_frame_01"], 0.0, 2.0, room_d * 0.5 - 0.05)

    if has("outside_view_01"):
        set_pos(by_id["outside_view_01"], 0.0, 2.0, room_d * 0.5 + 0.4)
        set_rot_y(by_id["outside_view_01"], 180.0)

    # Apply explicit constraints.
    for c in constraints.get("constraints", []):
        source_id = c.get("source")
        target_id = c.get("target")
        relation = c.get("relation")
        if source_id not in by_id or target_id not in by_id:
            report["warnings"].append(f"Skipped constraint with missing object: {c}")
            continue
        s = by_id[source_id]
        t = by_id[target_id]

        if relation == "on":
            tx, _, tz = get_pos(t)
            _, th, _ = approx_size(t)
            _, _, _ = approx_size(s)
            set_pos(s, tx, th, tz)
            report["steps"].append(f"Applied on: {source_id} on {target_id}")

        elif relation == "face to":
            set_rot_y(s, yaw_face_to(s, t))
            report["steps"].append(f"Applied face to: {source_id} faces {target_id}")

        elif relation == "behind":
            tx, ty, tz = get_pos(t)
            set_pos(s, tx, ty, tz + 0.45)
            report["steps"].append(f"Applied behind: {source_id} behind {target_id}")

        elif relation == "against":
            # Use target wall ID as semantic hint.
            if "left" in target_id:
                sx, _, _ = approx_size(s)
                x, y, z = get_pos(s)
                set_pos(s, -room_w * 0.5 + sx * 0.5 + 0.1, y, z)
                set_rot_y(s, 90.0)
            elif "right" in target_id:
                sx, _, _ = approx_size(s)
                x, y, z = get_pos(s)
                set_pos(s, room_w * 0.5 - sx * 0.5 - 0.1, y, z)
                set_rot_y(s, -90.0)
            elif "back" in target_id:
                _, _, sz = approx_size(s)
                x, y, z = get_pos(s)
                set_pos(s, x, y, room_d * 0.5 - sz * 0.5 - 0.1)
                set_rot_y(s, 180.0)

    # Clamp floor objects to room.
    for obj in by_id.values():
        if obj.get("representation") == "mesh" and not obj.get("background_only", False):
            clamp_to_room(obj, room_size)  # type: ignore

    # Simple overlap report. Real correction should happen in Unity with true bounds.
    movable = [
        obj
        for obj in by_id.values()
        if obj.get("category") in {"sofa", "chair", "table", "bookshelf", "plant"}
    ]
    overlaps = []
    for i in range(len(movable)):
        for j in range(i + 1, len(movable)):
            if intersects_xz(movable[i], movable[j]):
                overlaps.append([movable[i]["id"], movable[j]["id"]])
    report["approx_overlaps_xz"] = overlaps
    if overlaps:
        report["warnings"].append("Approximate XZ overlaps remain; Unity bounds correction should resolve them.")

    out["auto_layout_method"] = "rule_based_constraints_v0"
    out["auto_layout_note"] = "Initial semantic layout. Use Unity SceneLayoutAutoPlacer for true Renderer.bounds correction."
    return out, report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--layout", required=True, help="Input layout_for_unity.json")
    parser.add_argument("--constraints", required=True, help="Input spatial_constraints.json")
    parser.add_argument("--output", required=True, help="Output layout_for_unity_auto.json")
    parser.add_argument("--report", default="", help="Optional placement report JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    layout = load_json(Path(args.layout))
    constraints = load_json(Path(args.constraints))
    solved, report = solve(layout, constraints)
    save_json(solved, Path(args.output))
    if args.report:
        save_json(report, Path(args.report))


if __name__ == "__main__":
    main()
