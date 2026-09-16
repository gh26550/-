#!/usr/bin/env python3
"""
Validate scene_graph.json for the Mesh / Gaussian scene representation policy.

Policy:
- Mesh:
  - architecture
  - furniture
  - interactable objects
  - collidable objects
  - walkable elements
  - movable objects
  - objects requiring accurate close observation
- Gaussian:
  - background_only == true
  - no interaction
  - no collision
  - not walkable
  - not movable
  - no close observation
  - no collider / rigidbody

Usage:
    python scripts/17_validate_scene_graph.py \
        --input configs/scene_graph_final.json

Optional:
    python scripts/17_validate_scene_graph.py \
        --input configs/scene_graph_final.json \
        --write_report outputs/validation/scene_graph_report.json
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any


ALLOWED_REPRESENTATIONS = {"mesh", "gaussian"}
REQUIRED_ROOT_KEYS = {
    "schema_version",
    "scene_id",
    "scene_name",
    "coordinate_system",
    "classification_policy",
    "groups",
    "objects",
}
REQUIRED_OBJECT_KEYS = {
    "id",
    "name",
    "group_id",
    "category",
    "semantic_role",
    "parent_id",
    "transform",
    "properties",
    "representation",
    "asset",
}
REQUIRED_PROPERTY_KEYS = {
    "interaction_required",
    "collision_required",
    "walkable",
    "movable",
    "close_observation_required",
    "background_only",
}
REQUIRED_TRANSFORM_KEYS = {
    "position",
    "rotation_euler",
    "scale",
}


@dataclass
class ValidationIssue:
    level: str
    code: str
    object_id: str | None
    message: str


class SceneGraphValidator:
    def __init__(self, data: dict[str, Any], source_path: Path) -> None:
        self.data = data
        self.source_path = source_path
        self.issues: list[ValidationIssue] = []

    def error(
        self,
        code: str,
        message: str,
        object_id: str | None = None,
    ) -> None:
        self.issues.append(
            ValidationIssue(
                level="error",
                code=code,
                object_id=object_id,
                message=message,
            )
        )

    def warning(
        self,
        code: str,
        message: str,
        object_id: str | None = None,
    ) -> None:
        self.issues.append(
            ValidationIssue(
                level="warning",
                code=code,
                object_id=object_id,
                message=message,
            )
        )

    def validate(self) -> bool:
        self._validate_root()
        self._validate_coordinate_system()
        self._validate_groups()
        self._validate_objects()
        return not any(issue.level == "error" for issue in self.issues)

    def _validate_root(self) -> None:
        if not isinstance(self.data, dict):
            self.error("ROOT_NOT_OBJECT", "The JSON root must be an object.")
            return

        missing = REQUIRED_ROOT_KEYS - set(self.data.keys())
        for key in sorted(missing):
            self.error("MISSING_ROOT_KEY", f"Missing root key: {key}")

        objects = self.data.get("objects")
        if objects is not None and not isinstance(objects, list):
            self.error("OBJECTS_NOT_LIST", "'objects' must be a list.")

        groups = self.data.get("groups")
        if groups is not None and not isinstance(groups, list):
            self.error("GROUPS_NOT_LIST", "'groups' must be a list.")

    def _validate_coordinate_system(self) -> None:
        coordinate_system = self.data.get("coordinate_system")
        if not isinstance(coordinate_system, dict):
            return

        expected = {
            "engine": "Unity",
            "handedness": "left",
            "up_axis": "Y",
            "forward_axis": "Z",
            "unit": "meter",
        }

        for key, expected_value in expected.items():
            actual = coordinate_system.get(key)
            if actual != expected_value:
                self.warning(
                    "COORDINATE_SYSTEM_MISMATCH",
                    f"coordinate_system.{key} is {actual!r}; expected "
                    f"{expected_value!r} for the current Unity pipeline.",
                )

    def _validate_groups(self) -> None:
        groups = self.data.get("groups", [])
        if not isinstance(groups, list):
            return

        seen_ids: set[str] = set()

        for index, group in enumerate(groups):
            if not isinstance(group, dict):
                self.error(
                    "GROUP_NOT_OBJECT",
                    f"groups[{index}] must be an object.",
                )
                continue

            group_id = group.get("id")
            if not isinstance(group_id, str) or not group_id.strip():
                self.error(
                    "INVALID_GROUP_ID",
                    f"groups[{index}].id must be a non-empty string.",
                )
                continue

            if group_id in seen_ids:
                self.error(
                    "DUPLICATE_GROUP_ID",
                    f"Duplicate group id: {group_id}",
                )
            seen_ids.add(group_id)

            policy = group.get("representation_policy")
            if policy not in ALLOWED_REPRESENTATIONS:
                self.error(
                    "INVALID_GROUP_POLICY",
                    f"Group '{group_id}' has invalid representation_policy "
                    f"{policy!r}.",
                )

    def _validate_objects(self) -> None:
        objects = self.data.get("objects", [])
        if not isinstance(objects, list):
            return

        group_map = {
            group.get("id"): group
            for group in self.data.get("groups", [])
            if isinstance(group, dict) and isinstance(group.get("id"), str)
        }

        object_ids: set[str] = set()

        for index, obj in enumerate(objects):
            if not isinstance(obj, dict):
                self.error(
                    "OBJECT_NOT_OBJECT",
                    f"objects[{index}] must be an object.",
                )
                continue

            object_id = obj.get("id")
            if not isinstance(object_id, str) or not object_id.strip():
                self.error(
                    "INVALID_OBJECT_ID",
                    f"objects[{index}].id must be a non-empty string.",
                )
                object_id = f"<objects[{index}]>"

            if object_id in object_ids:
                self.error(
                    "DUPLICATE_OBJECT_ID",
                    f"Duplicate object id: {object_id}",
                    object_id,
                )
            object_ids.add(object_id)

            self._validate_object_structure(obj, object_id)
            self._validate_object_group(obj, object_id, group_map)
            self._validate_transform(obj, object_id)
            self._validate_properties(obj, object_id)
            self._validate_representation_policy(obj, object_id)
            self._validate_asset(obj, object_id)
            self._validate_physics(obj, object_id)

        for obj in objects:
            if not isinstance(obj, dict):
                continue
            object_id = obj.get("id")
            parent_id = obj.get("parent_id")
            if parent_id is not None and parent_id not in object_ids:
                self.error(
                    "UNKNOWN_PARENT_ID",
                    f"Object '{object_id}' refers to unknown parent_id "
                    f"{parent_id!r}.",
                    object_id if isinstance(object_id, str) else None,
                )

    def _validate_object_structure(
        self,
        obj: dict[str, Any],
        object_id: str,
    ) -> None:
        missing = REQUIRED_OBJECT_KEYS - set(obj.keys())
        for key in sorted(missing):
            self.error(
                "MISSING_OBJECT_KEY",
                f"Object '{object_id}' is missing key: {key}",
                object_id,
            )

    def _validate_object_group(
        self,
        obj: dict[str, Any],
        object_id: str,
        group_map: dict[str, dict[str, Any]],
    ) -> None:
        group_id = obj.get("group_id")
        if group_id not in group_map:
            self.error(
                "UNKNOWN_GROUP_ID",
                f"Object '{object_id}' refers to unknown group_id "
                f"{group_id!r}.",
                object_id,
            )
            return

        representation = obj.get("representation")
        group_policy = group_map[group_id].get("representation_policy")
        if representation in ALLOWED_REPRESENTATIONS and group_policy != representation:
            self.error(
                "GROUP_POLICY_CONFLICT",
                f"Object '{object_id}' uses representation "
                f"{representation!r}, but group '{group_id}' requires "
                f"{group_policy!r}.",
                object_id,
            )

    def _validate_transform(
        self,
        obj: dict[str, Any],
        object_id: str,
    ) -> None:
        transform = obj.get("transform")
        if not isinstance(transform, dict):
            self.error(
                "INVALID_TRANSFORM",
                f"Object '{object_id}' transform must be an object.",
                object_id,
            )
            return

        missing = REQUIRED_TRANSFORM_KEYS - set(transform.keys())
        for key in sorted(missing):
            self.error(
                "MISSING_TRANSFORM_KEY",
                f"Object '{object_id}' transform is missing: {key}",
                object_id,
            )

        for key in REQUIRED_TRANSFORM_KEYS:
            value = transform.get(key)
            if value is None:
                continue

            if (
                not isinstance(value, list)
                or len(value) != 3
                or not all(
                    isinstance(number, (int, float))
                    and not isinstance(number, bool)
                    for number in value
                )
            ):
                self.error(
                    "INVALID_TRANSFORM_VECTOR",
                    f"Object '{object_id}' transform.{key} must be a "
                    f"3-number list.",
                    object_id,
                )

        scale = transform.get("scale")
        if (
            isinstance(scale, list)
            and len(scale) == 3
            and all(isinstance(v, (int, float)) for v in scale)
            and any(v == 0 for v in scale)
        ):
            self.error(
                "ZERO_SCALE",
                f"Object '{object_id}' has a zero scale component.",
                object_id,
            )

    def _validate_properties(
        self,
        obj: dict[str, Any],
        object_id: str,
    ) -> None:
        properties = obj.get("properties")
        if not isinstance(properties, dict):
            self.error(
                "INVALID_PROPERTIES",
                f"Object '{object_id}' properties must be an object.",
                object_id,
            )
            return

        missing = REQUIRED_PROPERTY_KEYS - set(properties.keys())
        for key in sorted(missing):
            self.error(
                "MISSING_PROPERTY_KEY",
                f"Object '{object_id}' properties is missing: {key}",
                object_id,
            )

        for key in REQUIRED_PROPERTY_KEYS:
            if key in properties and not isinstance(properties[key], bool):
                self.error(
                    "PROPERTY_NOT_BOOLEAN",
                    f"Object '{object_id}' properties.{key} must be boolean.",
                    object_id,
                )

    @staticmethod
    def expected_representation(properties: dict[str, Any]) -> str:
        gaussian_allowed = (
            properties.get("background_only") is True
            and properties.get("interaction_required") is False
            and properties.get("collision_required") is False
            and properties.get("walkable") is False
            and properties.get("movable") is False
            and properties.get("close_observation_required") is False
        )
        return "gaussian" if gaussian_allowed else "mesh"

    def _validate_representation_policy(
        self,
        obj: dict[str, Any],
        object_id: str,
    ) -> None:
        representation = obj.get("representation")
        if representation not in ALLOWED_REPRESENTATIONS:
            self.error(
                "INVALID_REPRESENTATION",
                f"Object '{object_id}' representation must be 'mesh' or "
                f"'gaussian', got {representation!r}.",
                object_id,
            )
            return

        properties = obj.get("properties")
        if not isinstance(properties, dict):
            return

        expected = self.expected_representation(properties)
        if representation != expected:
            self.error(
                "REPRESENTATION_POLICY_VIOLATION",
                f"Object '{object_id}' is '{representation}', but its "
                f"properties require '{expected}'. Gaussian is allowed only "
                f"for background-only objects with no interaction, collision, "
                f"walkability, movement, or close-observation requirement.",
                object_id,
            )

        semantic_role = obj.get("semantic_role")
        if representation == "gaussian" and semantic_role != "background":
            self.error(
                "GAUSSIAN_NOT_BACKGROUND",
                f"Object '{object_id}' is Gaussian but semantic_role is "
                f"{semantic_role!r}; it must be 'background'.",
                object_id,
            )

        if (
            representation == "mesh"
            and properties.get("background_only") is True
            and expected == "gaussian"
        ):
            self.warning(
                "BACKGROUND_STORED_AS_MESH",
                f"Object '{object_id}' satisfies the Gaussian background "
                f"conditions but is stored as Mesh.",
                object_id,
            )

    def _validate_asset(
        self,
        obj: dict[str, Any],
        object_id: str,
    ) -> None:
        asset = obj.get("asset")
        if not isinstance(asset, dict):
            self.error(
                "INVALID_ASSET",
                f"Object '{object_id}' asset must be an object.",
                object_id,
            )
            return

        path = asset.get("path")
        asset_type = asset.get("type")
        representation = obj.get("representation")

        if not isinstance(path, str) or not path.strip():
            self.error(
                "INVALID_ASSET_PATH",
                f"Object '{object_id}' asset.path must be a non-empty string.",
                object_id,
            )

        if representation == "mesh":
            if asset_type not in {"glb", "gltf", "fbx", "obj", "unity_mesh"}:
                self.error(
                    "MESH_ASSET_TYPE_MISMATCH",
                    f"Mesh object '{object_id}' has unsupported asset type "
                    f"{asset_type!r}.",
                    object_id,
                )

        if representation == "gaussian":
            if asset_type not in {
                "gaussian_splat_asset",
                "ply",
                "spz",
            }:
                self.error(
                    "GAUSSIAN_ASSET_TYPE_MISMATCH",
                    f"Gaussian object '{object_id}' has unsupported asset type "
                    f"{asset_type!r}.",
                    object_id,
                )

    def _validate_physics(
        self,
        obj: dict[str, Any],
        object_id: str,
    ) -> None:
        representation = obj.get("representation")
        physics = obj.get("physics")

        if representation == "gaussian":
            if not isinstance(physics, dict):
                self.warning(
                    "GAUSSIAN_PHYSICS_MISSING",
                    f"Gaussian object '{object_id}' has no explicit physics "
                    f"settings. It should explicitly disable physics.",
                    object_id,
                )
                return

            collider = physics.get("collider", {})
            rigidbody = physics.get("rigidbody", {})

            if collider.get("enabled") is not False:
                self.error(
                    "GAUSSIAN_COLLIDER_ENABLED",
                    f"Gaussian object '{object_id}' must not have a collider.",
                    object_id,
                )

            if rigidbody.get("enabled") is not False:
                self.error(
                    "GAUSSIAN_RIGIDBODY_ENABLED",
                    f"Gaussian object '{object_id}' must not have a rigidbody.",
                    object_id,
                )

        if representation == "mesh":
            properties = obj.get("properties", {})
            collision_required = properties.get("collision_required")

            if collision_required is True:
                if not isinstance(physics, dict):
                    self.error(
                        "MESH_PHYSICS_MISSING",
                        f"Mesh object '{object_id}' requires collision but "
                        f"has no physics block.",
                        object_id,
                    )
                    return

                collider = physics.get("collider")
                if not isinstance(collider, dict) or collider.get("enabled") is not True:
                    self.error(
                        "REQUIRED_COLLIDER_DISABLED",
                        f"Mesh object '{object_id}' requires collision but its "
                        f"collider is not enabled.",
                        object_id,
                    )

            rigidbody = (
                physics.get("rigidbody", {})
                if isinstance(physics, dict)
                else {}
            )
            collider = (
                physics.get("collider", {})
                if isinstance(physics, dict)
                else {}
            )

            if (
                rigidbody.get("enabled") is True
                and rigidbody.get("is_kinematic") is False
                and collider.get("type") == "mesh"
                and collider.get("convex") is not True
            ):
                self.error(
                    "DYNAMIC_NONCONVEX_MESH_COLLIDER",
                    f"Object '{object_id}' uses a dynamic Rigidbody with a "
                    f"non-convex MeshCollider. Unity requires a convex collider "
                    f"or simpler proxy colliders for this configuration.",
                    object_id,
                )

    def summary(self) -> dict[str, Any]:
        error_count = sum(issue.level == "error" for issue in self.issues)
        warning_count = sum(issue.level == "warning" for issue in self.issues)

        objects = self.data.get("objects", [])
        mesh_count = 0
        gaussian_count = 0

        if isinstance(objects, list):
            for obj in objects:
                if not isinstance(obj, dict):
                    continue
                if obj.get("representation") == "mesh":
                    mesh_count += 1
                elif obj.get("representation") == "gaussian":
                    gaussian_count += 1

        return {
            "source": str(self.source_path),
            "valid": error_count == 0,
            "error_count": error_count,
            "warning_count": warning_count,
            "object_count": len(objects) if isinstance(objects, list) else 0,
            "mesh_count": mesh_count,
            "gaussian_count": gaussian_count,
            "issues": [asdict(issue) for issue in self.issues],
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate a Mesh/Gaussian scene graph JSON file."
    )
    parser.add_argument(
        "--input",
        required=True,
        help="Path to scene_graph.json",
    )
    parser.add_argument(
        "--write_report",
        default="",
        help="Optional path for a JSON validation report.",
    )
    return parser.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Input JSON not found: {path}")

    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)

    if not isinstance(data, dict):
        raise ValueError("The scene graph root must be a JSON object.")

    return data


def print_report(report: dict[str, Any]) -> None:
    print("=" * 72)
    print("Scene graph validation")
    print("=" * 72)
    print(f"[INFO] source         : {report['source']}")
    print(f"[INFO] objects        : {report['object_count']}")
    print(f"[INFO] mesh objects   : {report['mesh_count']}")
    print(f"[INFO] gaussian       : {report['gaussian_count']}")
    print(f"[INFO] errors         : {report['error_count']}")
    print(f"[INFO] warnings       : {report['warning_count']}")
    print("-" * 72)

    if not report["issues"]:
        print("[OK] No validation issues were found.")
    else:
        for issue in report["issues"]:
            object_text = (
                f" object={issue['object_id']}"
                if issue["object_id"] is not None
                else ""
            )
            prefix = "[ERROR]" if issue["level"] == "error" else "[WARN]"
            print(
                f"{prefix} {issue['code']}{object_text}: "
                f"{issue['message']}"
            )

    print("-" * 72)
    if report["valid"]:
        print("[PASS] The scene graph satisfies the current policy.")
    else:
        print("[FAIL] Correct the errors before scene generation.")
    print("=" * 72)


def main() -> int:
    args = parse_args()
    input_path = Path(args.input).expanduser().resolve()

    try:
        data = load_json(input_path)
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 2

    if data.get("schema_version") == "2.0":
        sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
        from pipeline import validate
        try:
            report = validate(data)
        except (ValueError, KeyError) as exc:
            print(f"[FAIL] {exc}")
            return 1
        if args.write_report:
            from pipeline import write
            write(args.write_report, report)
        print("[PASS] schema 2.0")
        return 0
    validator = SceneGraphValidator(data, input_path)
    validator.validate()
    report = validator.summary()
    print_report(report)

    if args.write_report:
        report_path = Path(args.write_report).expanduser().resolve()
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with report_path.open("w", encoding="utf-8") as file:
            json.dump(report, file, ensure_ascii=False, indent=2)
        print(f"[SAVED] Validation report: {report_path}")

    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
