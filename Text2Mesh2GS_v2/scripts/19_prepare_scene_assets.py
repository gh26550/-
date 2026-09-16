#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import sys
from copy import deepcopy
from pathlib import Path
from typing import Any

UNITY_PREFIX = "Assets/"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--mesh_manifest", required=True)
    p.add_argument("--gaussian_manifest", required=True)
    p.add_argument("--project_root", default=".")
    p.add_argument("--output_dir", required=True)
    p.add_argument("--strict", action="store_true")
    return p.parse_args()


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"JSON not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)
    if not isinstance(data, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return data


def save_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def is_unity_path(value: str) -> bool:
    return value.replace("\\", "/").startswith(UNITY_PREFIX)


def resolve_local_path(value: str, project_root: Path) -> Path | None:
    if is_unity_path(value):
        return None
    p = Path(value).expanduser()
    return p.resolve() if p.is_absolute() else (project_root / p).resolve()


def get_source(obj: dict[str, Any], representation: str):
    asset = obj.get("asset", {})
    if not isinstance(asset, dict):
        return None, "missing"

    keys = ("source_path", "source_glb", "path") if representation == "mesh" else (
        "source_ply", "source_spz", "source_path", "path", "source_glb"
    )
    for key in keys:
        value = asset.get(key)
        if isinstance(value, str) and value.strip():
            return value, key

    generation = obj.get("generation", {})
    if isinstance(generation, dict):
        for key in ("input_glb", "input_scene", "input_images"):
            value = generation.get(key)
            if isinstance(value, str) and value.strip():
                return value, f"generation.{key}"

    return None, "missing"


def build_mesh_jobs(manifest, project_root):
    jobs, missing = [], []
    for obj in manifest.get("objects", []):
        if not isinstance(obj, dict):
            continue
        oid = str(obj.get("id", "unknown"))
        source_value, source_kind = get_source(obj, "mesh")
        unity_path, resolved, exists = None, None, False

        if source_value:
            if is_unity_path(source_value):
                unity_path = source_value.replace("\\", "/")
            else:
                rp = resolve_local_path(source_value, project_root)
                if rp is not None:
                    resolved = str(rp)
                    exists = rp.exists()

        status = "ready" if unity_path or exists else "missing_source"
        jobs.append({
            "job_id": f"mesh_{oid}",
            "object_id": oid,
            "name": obj.get("name"),
            "category": obj.get("category"),
            "transform": deepcopy(obj.get("transform")),
            "source_kind": source_kind,
            "source_path": resolved,
            "unity_asset_path": unity_path,
            "source_exists": exists if resolved else None,
            "asset_type": obj.get("asset", {}).get("type") if isinstance(obj.get("asset"), dict) else None,
            "physics": deepcopy(obj.get("physics")),
            "affordances": deepcopy(obj.get("affordances", [])),
            "status": status
        })
        if status == "missing_source":
            missing.append({
                "representation": "mesh",
                "object_id": oid,
                "requested_source": source_value
            })
    return jobs, missing


def gaussian_status(source_kind, source_value):
    if not source_value:
        return "missing_source"
    v = source_value.lower()
    if source_kind == "path" and is_unity_path(source_value) and v.endswith(".asset"):
        return "unity_asset_ready"
    if v.endswith(".ply") or source_kind == "source_ply":
        return "ply_ready"
    if v.endswith(".spz") or source_kind == "source_spz":
        return "spz_ready"
    if v.endswith(".glb") or source_kind in {"source_glb", "generation.input_glb", "generation.input_scene"}:
        return "needs_gaussian_conversion"
    if source_kind == "generation.input_images":
        return "needs_gaussian_training"
    return "unknown_source_type"


def build_gaussian_jobs(manifest, project_root):
    jobs, missing = [], []
    for obj in manifest.get("objects", []):
        if not isinstance(obj, dict):
            continue
        oid = str(obj.get("id", "unknown"))
        source_value, source_kind = get_source(obj, "gaussian")
        unity_path, resolved, exists = None, None, False

        if source_value:
            if is_unity_path(source_value):
                unity_path = source_value.replace("\\", "/")
            else:
                rp = resolve_local_path(source_value, project_root)
                if rp is not None:
                    resolved = str(rp)
                    exists = rp.exists()

        status = gaussian_status(source_kind, source_value)
        if resolved and not exists:
            status = "missing_source"

        generation = obj.get("generation", {})
        if not isinstance(generation, dict):
            generation = {}

        jobs.append({
            "job_id": f"gaussian_{oid}",
            "object_id": oid,
            "name": obj.get("name"),
            "category": obj.get("category"),
            "semantic_role": obj.get("semantic_role"),
            "transform": deepcopy(obj.get("transform")),
            "source_kind": source_kind,
            "source_path": resolved,
            "unity_asset_path": unity_path,
            "source_exists": exists if resolved else None,
            "status": status,
            "generation": {
                "multiview_output_dir": generation.get("multiview_output_dir", f"outputs/multiview/{oid}"),
                "gaussian_output_dir": generation.get("gaussian_output_dir", f"outputs/gaussian/{oid}"),
                "num_azimuth_views": generation.get("num_azimuth_views", 24),
                "elevations": generation.get("elevations", [0, 20]),
                "num_points": generation.get("num_points", 120000),
                "steps": generation.get("steps", 9000),
                "output_ply": generation.get("output_ply", f"outputs/gaussian/{oid}/point_cloud.ply")
            },
            "physics_policy": {"collider": False, "rigidbody": False}
        })
        if status == "missing_source":
            missing.append({
                "representation": "gaussian",
                "object_id": oid,
                "requested_source": source_value
            })
    return jobs, missing


def main():
    args = parse_args()
    mesh_path = Path(args.mesh_manifest).expanduser().resolve()
    gaussian_path = Path(args.gaussian_manifest).expanduser().resolve()
    project_root = Path(args.project_root).expanduser().resolve()
    output_dir = Path(args.output_dir).expanduser().resolve()

    try:
        mesh_manifest = load_json(mesh_path)
        gaussian_manifest = load_json(gaussian_path)
    except Exception as exc:
        print(f"[FATAL] {exc}", file=sys.stderr)
        return 2

    mesh_jobs, mesh_missing = build_mesh_jobs(mesh_manifest, project_root)
    gaussian_jobs, gaussian_missing = build_gaussian_jobs(gaussian_manifest, project_root)
    missing = mesh_missing + gaussian_missing

    mesh_jobs_path = output_dir / "mesh_import_jobs.json"
    gaussian_jobs_path = output_dir / "gaussian_import_jobs.json"
    summary_path = output_dir / "asset_preparation_summary.json"

    save_json(mesh_jobs_path, {
        "scene_id": mesh_manifest.get("scene_id"),
        "job_type": "mesh_import_jobs",
        "job_count": len(mesh_jobs),
        "jobs": mesh_jobs
    })
    save_json(gaussian_jobs_path, {
        "scene_id": gaussian_manifest.get("scene_id"),
        "job_type": "gaussian_import_jobs",
        "job_count": len(gaussian_jobs),
        "jobs": gaussian_jobs
    })
    save_json(summary_path, {
        "scene_id": mesh_manifest.get("scene_id") or gaussian_manifest.get("scene_id"),
        "project_root": str(project_root),
        "mesh_job_count": len(mesh_jobs),
        "gaussian_job_count": len(gaussian_jobs),
        "missing_source_count": len(missing),
        "missing_sources": missing,
        "files": {
            "mesh_import_jobs": str(mesh_jobs_path),
            "gaussian_import_jobs": str(gaussian_jobs_path)
        }
    })

    print("=" * 72)
    print("Scene asset preparation")
    print("=" * 72)
    print(f"[INFO] mesh jobs       : {len(mesh_jobs)}")
    print(f"[INFO] gaussian jobs   : {len(gaussian_jobs)}")
    print(f"[INFO] missing sources : {len(missing)}")
    print(f"[SAVED] {mesh_jobs_path}")
    print(f"[SAVED] {gaussian_jobs_path}")
    print(f"[SAVED] {summary_path}")
    print("=" * 72)

    if args.strict and missing:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
