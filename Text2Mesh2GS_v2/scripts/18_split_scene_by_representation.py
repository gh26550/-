#!/usr/bin/env python3
"""
Split a validated scene_graph.json into Mesh and Gaussian manifests.

Outputs:
- mesh_manifest.json
- gaussian_manifest.json
- scene_split_summary.json

This script does not modify source assets. It only creates manifests that can
be consumed by later export / generation scripts.

Usage:
    python scripts/18_split_scene_by_representation.py \
        --input configs/scene_graph_final.json \
        --output_dir outputs/scene_split/room_001

Recommended workflow:
    1. Validate scene_graph.json with 17_validate_scene_graph.py
    2. Run this script
    3. Use mesh_manifest.json for Mesh export / Unity loading
    4. Use gaussian_manifest.json for Gaussian background generation
"""

from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any


ALLOWED_REPRESENTATIONS = {"mesh", "gaussian"}

ROOT_METADATA_KEYS = [
    "schema_version",
    "scene_id",
    "scene_name",
    "description",
    "coordinate_system",
    "scene_bounds",
]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Split scene graph objects into Mesh and Gaussian manifests."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to the source scene_graph.json.",
    )
    parser.add_argument(
        "--output_dir",
        required=True,
        help="Directory where split manifests will be written.",
    )
    parser.add_argument(
        "--strict",
        action="store_true",
        help=(
            "Fail if an object violates the Mesh/Gaussian background policy. "
            "Recommended after validation."
        ),
    )
    parser.add_argument(
        "--copy_asset_paths_as_absolute",
        action="store_true",
        help=(
            "Convert local non-Unity asset paths to absolute paths when possible. "
            "Unity Assets/... paths remain unchanged."
        ),
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSON not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("The scene graph root must be a JSON object.")

    objects = data.get("objects")
    if not isinstance(objects, list):
        raise ValueError("'objects' must be a list.")

    return data


def gaussian_allowed(properties: dict[str, Any]) -> bool:
    return (
        properties.get("background_only") is True
        and properties.get("interaction_required") is False
        and properties.get("collision_required") is False
        and properties.get("walkable") is False
        and properties.get("movable") is False
        and properties.get("close_observation_required") is False
    )


def expected_representation(obj: dict[str, Any]) -> str:
    properties = obj.get("properties", {})
    if isinstance(properties, dict) and gaussian_allowed(properties):
        return "gaussian"
    return "mesh"


def normalize_asset_path(
    asset_path: str,
    source_json_path: Path,
    make_absolute: bool,
) -> str:
    if not make_absolute:
        return asset_path

    # Unity project-relative paths should stay as-is.
    normalized = asset_path.replace("\\", "/")
    if normalized.startswith("Assets/"):
        return normalized

    path = Path(asset_path).expanduser()
    if path.is_absolute():
        return str(path.resolve())

    return str((source_json_path.parent / path).resolve())


def prepare_object(
    obj: dict[str, Any],
    source_json_path: Path,
    make_absolute: bool,
) -> dict[str, Any]:
    result = deepcopy(obj)

    asset = result.get("asset")
    if isinstance(asset, dict):
        path = asset.get("path")
        if isinstance(path, str) and path.strip():
            asset["path"] = normalize_asset_path(
                path,
                source_json_path,
                make_absolute,
            )

        source_ply = asset.get("source_ply")
        if isinstance(source_ply, str) and source_ply.strip():
            asset["source_ply"] = normalize_asset_path(
                source_ply,
                source_json_path,
                make_absolute,
            )

    return result


def build_manifest(
    source_data: dict[str, Any],
    representation: str,
    objects: list[dict[str, Any]],
    source_path: Path,
) -> dict[str, Any]:
    manifest: dict[str, Any] = {}

    for key in ROOT_METADATA_KEYS:
        if key in source_data:
            manifest[key] = deepcopy(source_data[key])

    manifest["manifest_type"] = f"{representation}_manifest"
    manifest["representation"] = representation
    manifest["source_scene_graph"] = str(source_path)
    manifest["object_count"] = len(objects)
    manifest["objects"] = objects

    if representation == "mesh":
        manifest["usage"] = {
            "purpose": (
                "Architecture, furniture, collision, locomotion, "
                "and user-interactable objects."
            ),
            "unity_components": [
                "MeshFilter",
                "MeshRenderer",
                "Collider",
                "Rigidbody_when_required",
                "XR_Interactable_when_required",
            ],
        }
    else:
        manifest["usage"] = {
            "purpose": (
                "Non-interactable and non-collidable background only."
            ),
            "unity_components": [
                "GaussianSplatRenderer",
            ],
            "physics_policy": {
                "collider": False,
                "rigidbody": False,
            },
        }

    return manifest


def validate_for_split(
    objects: list[Any],
    strict: bool,
) -> tuple[list[dict[str, Any]], list[str]]:
    valid_objects: list[dict[str, Any]] = []
    messages: list[str] = []
    seen_ids: set[str] = set()

    for index, obj in enumerate(objects):
        if not isinstance(obj, dict):
            message = f"objects[{index}] is not an object."
            if strict:
                raise ValueError(message)
            messages.append(f"[SKIP] {message}")
            continue

        object_id = obj.get("id")
        if not isinstance(object_id, str) or not object_id.strip():
            message = f"objects[{index}] has an invalid id."
            if strict:
                raise ValueError(message)
            messages.append(f"[SKIP] {message}")
            continue

        if object_id in seen_ids:
            message = f"Duplicate object id: {object_id}"
            if strict:
                raise ValueError(message)
            messages.append(f"[SKIP] {message}")
            continue
        seen_ids.add(object_id)

        representation = obj.get("representation")
        if representation not in ALLOWED_REPRESENTATIONS:
            message = (
                f"Object '{object_id}' has invalid representation "
                f"{representation!r}."
            )
            if strict:
                raise ValueError(message)
            messages.append(f"[SKIP] {message}")
            continue

        expected = expected_representation(obj)
        if representation != expected:
            message = (
                f"Object '{object_id}' is marked '{representation}', "
                f"but its properties require '{expected}'."
            )
            if strict:
                raise ValueError(message)
            messages.append(f"[WARN] {message}")

        if representation == "gaussian":
            semantic_role = obj.get("semantic_role")
            if semantic_role != "background":
                message = (
                    f"Gaussian object '{object_id}' must use "
                    f"semantic_role='background'."
                )
                if strict:
                    raise ValueError(message)
                messages.append(f"[WARN] {message}")

        valid_objects.append(obj)

    return valid_objects, messages


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(data, file, ensure_ascii=False, indent=2)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    try:
        data = load_json(input_path)
        valid_objects, messages = validate_for_split(
            data["objects"],
            strict=args.strict,
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 2

    mesh_objects: list[dict[str, Any]] = []
    gaussian_objects: list[dict[str, Any]] = []

    for obj in valid_objects:
        prepared = prepare_object(
            obj,
            source_json_path=input_path,
            make_absolute=args.copy_asset_paths_as_absolute,
        )

        representation = prepared["representation"]
        if representation == "mesh":
            mesh_objects.append(prepared)
        elif representation == "gaussian":
            gaussian_objects.append(prepared)

    mesh_manifest = build_manifest(
        source_data=data,
        representation="mesh",
        objects=mesh_objects,
        source_path=input_path,
    )
    gaussian_manifest = build_manifest(
        source_data=data,
        representation="gaussian",
        objects=gaussian_objects,
        source_path=input_path,
    )

    mesh_path = output_dir / "mesh_manifest.json"
    gaussian_path = output_dir / "gaussian_manifest.json"
    summary_path = output_dir / "scene_split_summary.json"

    summary = {
        "source_scene_graph": str(input_path),
        "output_dir": str(output_dir),
        "scene_id": data.get("scene_id"),
        "scene_name": data.get("scene_name"),
        "total_objects": len(valid_objects),
        "mesh_count": len(mesh_objects),
        "gaussian_count": len(gaussian_objects),
        "mesh_object_ids": [obj["id"] for obj in mesh_objects],
        "gaussian_object_ids": [obj["id"] for obj in gaussian_objects],
        "messages": messages,
        "files": {
            "mesh_manifest": str(mesh_path),
            "gaussian_manifest": str(gaussian_path),
        },
    }

    save_json(mesh_path, mesh_manifest)
    save_json(gaussian_path, gaussian_manifest)
    save_json(summary_path, summary)

    print("=" * 72)
    print("Scene graph split")
    print("=" * 72)
    print(f"[INFO] source          : {input_path}")
    print(f"[INFO] scene id        : {data.get('scene_id')}")
    print(f"[INFO] total objects   : {len(valid_objects)}")
    print(f"[INFO] mesh objects    : {len(mesh_objects)}")
    print(f"[INFO] gaussian objects: {len(gaussian_objects)}")
    print("-" * 72)

    for message in messages:
        print(message)

    print(f"[SAVED] {mesh_path}")
    print(f"[SAVED] {gaussian_path}")
    print(f"[SAVED] {summary_path}")
    print("=" * 72)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
