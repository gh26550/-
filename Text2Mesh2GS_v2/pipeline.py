#!/usr/bin/env python3
"""Validated data bridge and Gaussian runner. No ML imports for prepare/validate/plan."""
from __future__ import annotations

import argparse
import copy
import csv
import json
import math
import hashlib
import subprocess
import sys
from pathlib import Path

VERSION = "2.0"
PROCEDURAL = {"floor", "ceiling", "wall", "window_frame", "rug"}
RELATIONS = {"on", "near", "face to", "against", "left of", "right of", "in front of",
             "behind", "side of", "center of room", "near room center", "center of scene",
             "center aligned with", "aligned with", "parallel to", "cover", "visible through"}
FURNITURE = {"on", "near", "face to", "against", "left of", "right of", "in front of",
             "behind", "side of", "center of room", "near room center", "center of scene"}
CENTER = {"center of room", "near room center", "center of scene"}


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def vector(value, name, positive=False):
    if not isinstance(value, list) or len(value) != 3 or any(
        isinstance(x, bool) or not isinstance(x, (int, float)) or not math.isfinite(x)
        or (positive and x <= 0) for x in value
    ):
        raise ValueError(f"{name}: expected three finite {'positive ' if positive else ''}numbers")
    return value


def normalise(raw, constraint_file=None):
    """Convert legacy simple/detailed graphs and Unity layouts without inferring coordinates."""
    scene = copy.deepcopy(raw)
    if constraint_file and constraint_file.get('scene_fingerprint'):
        digest = hashlib.sha256(json.dumps(raw, sort_keys=True, ensure_ascii=False).encode()).hexdigest()
        if digest != constraint_file['scene_fingerprint']:
            raise ValueError('Scene changed after constraint generation; rerun stage 34')
    objects = scene.get("objects", scene.get("unity_objects", []))
    if not objects:
        raise ValueError("objects/unity_objects is empty")
    result = {"schema_version": VERSION, "scene_id": scene.get("scene_id", ""),
              "scene_name": scene.get("scene_name", ""),
              "coordinate_system": "room-local: x=right, y=up, front=-z; units=metres",
              "room": copy.deepcopy(scene.get("room", {"size": [6, 4, 6]})), "objects": [],
              "constraints": copy.deepcopy((constraint_file or scene).get("constraints", []))}
    for key in ("reference_image", "generation"):
        if key in scene:
            result[key] = copy.deepcopy(scene[key])
    if constraint_file:
        result["constraint_generation"] = {k: copy.deepcopy(v) for k, v in constraint_file.items()
                                           if k not in {"constraints", "scene_id", "schema_version"}}
    elif "constraint_generation" in scene:
        result["constraint_generation"] = copy.deepcopy(scene["constraint_generation"])
    if constraint_file and constraint_file.get("scene_id", result["scene_id"]) != result["scene_id"]:
        raise ValueError("Constraint and layout scene_id differ")
    if constraint_file and "reference_image" in constraint_file:
        if constraint_file["reference_image"].get("sha256") != scene.get("reference_image", {}).get("sha256"):
            raise ValueError("Constraints were generated from a different reference image")
    for source in objects:
        obj = copy.deepcopy(source)
        props = obj.get("properties", {})
        background = props.get("background_only", obj.get("background_only", False))
        interactive = props.get("interaction_required", obj.get("interaction_required", False))
        rep = obj.get("representation", "mesh")
        category = obj.get("category", "")
        obj["properties"] = {
            "background_only": background,
            "interaction_required": interactive,
            "collision_required": props.get("collision_required", rep == "mesh" and category != "rug"),
            "walkable": props.get("walkable", category == "floor"),
            "movable": props.get("movable", rep == "mesh" and category not in PROCEDURAL),
            "close_observation_required": props.get("close_observation_required", interactive),
        }
        obj["semantic_role"] = "background" if rep == "gaussian" else obj.get("semantic_role", "physical")
        obj["interaction_required"] = interactive
        obj["background_only"] = background
        obj.setdefault("name", obj.get("id", ""))
        obj.setdefault("transform", {"position": [0, 0, 0], "rotation_euler": [0, 0, 0], "scale": [1, 1, 1]})
        # Declared dimensions are optional; never pretend category defaults are measured mesh size.
        obj.setdefault("forward_axis", [0, 0, 1])
        old_asset = obj.get("asset", {}) or {}
        unity_type = "gaussian_splat" if rep == "gaussian" else "procedural_mesh" if category in PROCEDURAL else "mesh_gameobject"
        if scene.get("schema_version") == VERSION or scene.get("generation", {}).get("method") == "local_vision":
            unity_type = old_asset.get("unity_type", unity_type)
        if unity_type not in {"gaussian_splat", "procedural_mesh", "mesh_gameobject"}:
            raise ValueError("Invalid unity_type")
        ext = "ply" if rep == "gaussian" else "glb"
        kind = "gaussian_assets" if rep == "gaussian" else "mesh_assets"
        expected = None if unity_type == "procedural_mesh" else f"outputs/{kind}/{result['scene_id']}/{obj['id']}/{obj['id']}.{ext}"
        obj["asset"] = {**old_asset, "unity_type": unity_type, "asset_path": expected,
                        "source_path": old_asset["source_path"] if "source_path" in old_asset else old_asset.get("source_glb") or old_asset.get("source_ply") or old_asset.get("asset_path") or old_asset.get("path"),
                        "prefab_name": obj["id"], "renderer_name": obj["id"]}
        result["objects"].append(obj)
    unique = {}
    for c in result["constraints"]:
        c["relation"] = str(c.get("relation", "")).strip().lower()
        c.setdefault("weight", 1.0)
        c.setdefault("type", "semantic")
        key = (c.get("source"), c["relation"], c.get("target"))
        if key in unique and unique[key]["weight"] != c["weight"]:
            raise ValueError(f"Conflicting weights on duplicate constraint {key}")
        unique[key] = c
    result["constraints"] = list(unique.values())
    validate(result)
    return result


def support_order(scene):
    ids = {o["id"] for o in scene["objects"] if o["properties"]["movable"]}
    parent = {}
    for c in scene["constraints"]:
        if c["relation"] == "on" and c["source"] in ids:
            if c["source"] in parent and parent[c["source"]] != c["target"]:
                raise ValueError(f"Multiple supports for {c['source']}")
            parent[c["source"]] = c["target"]
    order, visiting = [], set()
    def visit(oid):
        if oid not in ids or oid in order:
            return
        if oid in visiting:
            raise ValueError("Cyclic on constraints")
        visiting.add(oid)
        visit(parent.get(oid))
        visiting.remove(oid)
        order.append(oid)
    for oid in sorted(ids):
        visit(oid)
    return order


def validate(scene):
    if scene.get("schema_version") != VERSION or not scene.get("scene_id"):
        raise ValueError("Expected schema_version=2.0 and nonempty scene_id")
    if not isinstance(scene["scene_id"], str) or any(c in scene["scene_id"] for c in "/\\") or scene["scene_id"] in {".", ".."}:
        raise ValueError("Unsafe scene_id")
    vector(scene["room"]["size"], "room.size", True)
    ids = {}
    for obj in scene["objects"]:
        oid = obj.get("id")
        if not isinstance(oid, str) or not oid or oid in ids or any(c in oid for c in "/\\") or oid in {".", ".."}:
            raise ValueError(f"Missing, duplicate or unsafe object ID: {oid}")
        ids[oid] = obj
        for key in ("position", "rotation_euler", "scale"):
            vector(obj["transform"][key], f"{oid}.{key}", key == "scale")
        if obj["representation"] not in {"mesh", "gaussian"}:
            raise ValueError(f"Invalid representation: {oid}")
        props = obj["properties"]
        if any(not isinstance(v, bool) for v in props.values()):
            raise ValueError(f"Properties must be booleans: {oid}")
        if obj["representation"] == "gaussian" and (not props["background_only"] or any(props[k] for k in props if k != "background_only")):
            raise ValueError(f"Gaussian must be nonphysical background: {oid}")
        axis = vector(obj["forward_axis"], f"{oid}.forward_axis")
        if sum(x*x for x in axis) < 1e-12:
            raise ValueError(f"Zero forward_axis: {oid}")
    relations = set()
    for c in scene["constraints"]:
        rel, src, dst = c["relation"], c.get("source"), c.get("target")
        if rel not in RELATIONS or src not in ids or (dst not in ids and rel not in CENTER):
            raise ValueError(f"Unsupported relation or missing object: {c}")
        if src == dst:
            raise ValueError(f"Self constraint: {c}")
        weight = c["weight"]
        if isinstance(weight, bool) or not isinstance(weight, (int, float)) or not math.isfinite(weight) or weight < 0:
            raise ValueError(f"Invalid weight: {c}")
        if ids[src]["properties"]["movable"] and rel not in FURNITURE:
            raise ValueError(f"Furniture solver does not support {rel}")
        relations.add((src, rel, dst))
    for src, rel, dst in relations:
        if rel == 'against' and ids[dst]['category'] != 'wall':
            raise ValueError('against requires a wall target')
        if rel == 'against' and dst == 'wall_left_01' and (src, 'against', 'wall_right_01') in relations:
            raise ValueError('Contradictory contact with opposite walls: ' + src)
        opposite = {"left of": "right of", "right of": "left of", "behind": "in front of", "in front of": "behind"}.get(rel)
        if opposite and ((src, opposite, dst) in relations or (dst, rel, src) in relations):
            raise ValueError(f"Contradictory relations: {src} {rel} {dst}")
    support_order(scene)
    return {"valid": True, "objects": len(ids), "constraints": len(scene["constraints"])}


def export_bundle(scene, out, root):
    out, root = Path(out), Path(root).resolve()
    validate(scene)
    generation = scene.get('generation', {})
    if generation.get('review_pipeline') == 'per_object_v1':
        if generation.get('review_status') != 'complete' or generation.get('unresolved'):
            raise ValueError('Inventory has unresolved visual review items; export blocked')
        cg = scene.get('constraint_generation', {})
        if cg.get('status') != 'complete' or cg.get('unresolved'):
            raise ValueError('Complete reviewed constraints are required; export blocked')
        uncovered = [o['id'] for o in scene['objects'] if
                     (o['properties']['movable'] or o['representation'] == 'gaussian') and
                     not any(c['source'] == o['id'] for c in scene['constraints'])]
        if uncovered:
            raise ValueError('Missing constraints: ' + str(uncovered))
    unity = []
    import_jobs = []
    for obj in scene["objects"]:
        item = {k: copy.deepcopy(obj[k]) for k in ("id", "name", "category", "representation", "interaction_required", "background_only", "transform", "asset", "forward_axis")}
        item["movable"] = obj["properties"]["movable"]
        unity.append(item)
        if obj["representation"] != "gaussian":
            continue
        raw = obj["asset"].get("source_path")
        source = (root / raw).resolve() if raw else None
        status = "missing_source" if not source or not source.exists() else "ply_ready" if source.suffix.lower() == ".ply" else "needs_gaussian_conversion" if source.suffix.lower() == ".glb" else "unsupported_source"
        base = f"outputs/gaussian_assets/{scene['scene_id']}/{obj['id']}"
        import_jobs.append({"object_id": obj["id"], "source_path": str(source) if source else None, "status": status,
                            "generation": {"multiview_output_dir": base + "/multiview", "gaussian_output_dir": base + "/training", "output_ply": obj["asset"]["asset_path"]}})
    write(out / "scene.json", scene)
    write(out / "fixed_anchor_proposals.json", {'scene_id': scene['scene_id'],
          'warning': 'Estimated design positions, not measured geometry. Review in Unity.',
          'objects': [{k: copy.deepcopy(o[k]) for k in ('id', 'transform', 'dimensions_m', 'fixed_placement_proposal')}
                      for o in scene['objects'] if o.get('fixed_placement_proposal')]})
    write(out / "layout_for_unity.json", {"schema_version": VERSION, "scene_id": scene["scene_id"], "room": scene["room"], "unity_objects": unity})
    write(out / "spatial_constraints.json", {"schema_version": VERSION, "scene_id": scene["scene_id"], "constraints": scene["constraints"]})
    write(out / "gaussian_import_jobs.json", {"schema_version": VERSION, "scene_id": scene["scene_id"], "jobs": import_jobs})
    # Both import and generation formats derive from the same canonical source.
    for rep in ("mesh", "gaussian"):
        selected = [o for o in scene["objects"] if o["representation"] == rep]
        write(out / f"{rep}_manifest.json", {"schema_version": VERSION, "scene_id": scene["scene_id"], "objects": selected})
        jobs = [{"object_id": o["id"], "name": o["name"], "category": o["category"], "transform": o["transform"],
                 "object_prompt": o.get("asset_prompt", o["name"]), "background_prompt": o.get("asset_prompt", o["name"]),
                 "interaction_required": o["interaction_required"], "background_only": o["background_only"],
                 "expected_outputs": {"glb" if rep == "mesh" else "ply": o["asset"]["asset_path"]}}
                for o in selected]
        write(out / f"{rep}_jobs.json", {"scene_id": scene["scene_id"], "jobs": jobs})
    write(out / "validation.json", {**validate(scene), "support_order": support_order(scene), "assets_generated": False})
    # The VLM supplies the content; these are format adapters, not category-based prompt rules.
    mesh_assets = [o for o in scene["objects"] if o["asset"]["unity_type"] == "mesh_gameobject"]
    with (out / "mesh_object_prompts.csv").open("w", newline="", encoding="utf-8-sig") as stream:
        fields = ["scene_id", "object_id", "name", "category", "object_prompt", "expected_glb", "output_dir"]
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for o in mesh_assets:
            writer.writerow({"scene_id": scene["scene_id"], "object_id": o["id"], "name": o["name"],
                             "category": o["category"], "object_prompt": o.get("asset_prompt", o["name"]),
                             "expected_glb": o["asset"]["asset_path"], "output_dir": str(Path(o["asset"]["asset_path"]).parent)})
    write(out / "gaussian_background_prompts.json", {"scene_id": scene["scene_id"],
          "negative_prompt": "text, watermark, indoor foreground furniture",
          "prompts": [{"object_id": o["id"], "background_prompt": o.get("asset_prompt", o["name"])}
                      for o in scene["objects"] if o["representation"] == "gaussian"]})


def check_assets(scene, root):
    validate(scene)
    root = Path(root).resolve()
    rows = []
    for obj in scene["objects"]:
        if obj["asset"]["unity_type"] == "procedural_mesh":
            continue
        target = root / obj["asset"]["asset_path"]
        rows.append({"object_id": obj["id"], "path": str(target), "exists": target.is_file()})
    return {"ready": all(row["exists"] for row in rows), "assets": rows}


def gaussian_plan(jobs, config, project_root):
    root = Path(project_root).resolve()
    script_root = Path(__file__).parent / "scripts"
    python, blender = config.get("python", sys.executable), config.get("blender", "blender")
    render_script = config.get("multiview_script")
    render_script = str((root / render_script).resolve()) if render_script else str(script_root / "05_render_multiview_blender_depth.py")
    steps = int(config.get("steps", 1000))
    if steps <= 0 or int(config.get("num_points", 10000)) <= 0:
        raise ValueError("steps and num_points must be positive")
    lo, hi = float(config.get("scale_min", .00005)), float(config.get("scale_max", .005))
    if not 0 < lo <= hi or not math.isfinite(hi):
        raise ValueError("Invalid scale range")
    plan = []
    for job in jobs["jobs"]:
        if job["status"] == "ply_ready":
            # Already generated files must be explicitly staged at expected asset_path.
            plan.append({"object_id": job["object_id"], "status": "ready", "source": job["source_path"],
                         "expected_ply": str(root / job["generation"]["output_ply"]), "commands": []})
            continue
        if job["status"] != "needs_gaussian_conversion":
            raise ValueError(f"{job['object_id']}: {job['status']}; provide an existing GLB")
        gen = job["generation"]
        mv, output, ply = [str(root / gen[k]) for k in ("multiview_output_dir", "gaussian_output_dir", "output_ply")]
        ckpt = str(Path(output) / f"ckpt_{steps:06d}.pt")
        shared = ["--scale_min", str(lo), "--scale_max", str(hi)]
        render_settings = ["--background_color", config.get("background_color", "0.770588,0.771569,0.770588"), "--image_downscale", str(config.get("image_downscale", 2))]
        if config.get("use_zbuffer_visibility", False):
            render_settings += ["--use_zbuffer_visibility", "--zbuffer_depth_threshold", str(config.get("zbuffer_depth_threshold", .01))]
        train = [python, str(script_root / "07_5_train_gsplat_zbuffer_visibility.py"), "--multiview_dir", mv,
                 "--mesh_npz", str(Path(mv) / "aligned_mesh/model_aligned_blender.npz"), "--output_dir", output,
                 "--steps", str(steps), "--num_points", str(config.get("num_points", 10000)), *shared, *render_settings]
        if config.get("use_anisotropic_init", True):
            train += ["--use_anisotropic_init"]
        if config.get("use_alpha_loss", True):
            train += ["--use_alpha_loss", "--alpha_weight", str(config.get("alpha_weight", 1.0))]
        train += ["--seed", str(config.get("seed", 12345))]
        commands = [[blender, "--background", "--python", render_script, "--", "--input", job["source_path"],
                     "--output_dir", mv, "--resolution", "1024", "--num_azimuth_views", "24", "--elevations", "0,20",
                     "--camera_type", "perspective", "--background", "transparent", "--save_camera_params", "--save_aligned_glb"], train]
        for view in config.get("validation_views", [0, 6, 12]):
            commands.append([python, str(script_root / "08_render_gsplat_checkpoint_zbuffer_fixed.py"), "--checkpoint", ckpt,
                             "--multiview_dir", mv, "--output_dir", str(Path(output) / "validation"), "--view_index", str(view), *shared, *render_settings])
            if config.get("use_zbuffer_visibility", False):
                commands.append([python, str(script_root / "08_render_gsplat_checkpoint_zbuffer_fixed.py"), "--checkpoint", ckpt,
                                 "--multiview_dir", mv, "--output_dir", str(Path(output) / "validation_portable"), "--view_index", str(view),
                                 *shared, *render_settings, "--no_zbuffer_visibility"])
        commands.append([python, str(script_root / "09_export_gaussian_ply.py"), "--checkpoint", ckpt, "--output", ply, *shared])
        plan.append({"object_id": job["object_id"], "status": "planned", "commands": commands,
                     "expected_ply": ply, "source": job["source_path"]})
    return plan


def execute(plan, root, output):
    """Stop on first failure, record every completed step; never shell-evaluate command text."""
    root, output = Path(root).resolve(), Path(output)
    results = []
    for job in plan:
        if job["status"] == "ready":
            if Path(job["source"]).resolve() != Path(job["expected_ply"]).resolve():
                raise ValueError(f"Stage existing PLY at expected path before execution: {job['expected_ply']}")
            continue
        if not Path(job["source"]).is_file():
            raise ValueError(f"Missing source: {job['source']}")
        for cmd in job["commands"]:
            script = cmd[cmd.index("--python")+1] if "--python" in cmd else cmd[1]
            if not Path(script).is_file():
                raise ValueError(f"Missing dependency script: {script}")
        for index, cmd in enumerate(job["commands"]):
            log = output / "logs" / job["object_id"] / f"step_{index:02d}.log"
            log.parent.mkdir(parents=True, exist_ok=True)
            try:
                with log.open("w", encoding="utf-8") as stream:
                    completed = subprocess.run(cmd, cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=False)
                rc = completed.returncode
            except OSError as exc:
                rc = -1
                log.write_text(str(exc), encoding="utf-8")
            results.append({"object_id": job["object_id"], "step": index, "return_code": rc, "log": str(log)})
            write(output / "run_results.json", results)
            if rc:
                raise RuntimeError(f"Step failed: {log}")
        if not Path(job["expected_ply"]).exists():
            raise RuntimeError(f"Expected output missing: {job['expected_ply']}")
    return results


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--scene", required=True)
    prep.add_argument("--constraints")
    prep.add_argument("--output", required=True)
    prep.add_argument("--project-root", default=".")
    check = sub.add_parser("validate")
    check.add_argument("--scene", required=True)
    assets = sub.add_parser("assets")
    assets.add_argument("--scene", required=True)
    assets.add_argument("--project-root", default=".")
    run = sub.add_parser("gaussian")
    run.add_argument("--jobs", required=True)
    run.add_argument("--config", required=True)
    run.add_argument("--project-root", default=".")
    run.add_argument("--output", required=True)
    run.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    if args.command == "prepare":
        scene = normalise(read(args.scene), read(args.constraints) if args.constraints else None)
        export_bundle(scene, args.output, args.project_root)
        print(json.dumps(validate(scene)))
    elif args.command == "validate":
        print(json.dumps(validate(read(args.scene))))
    elif args.command == "assets":
        report = check_assets(read(args.scene), args.project_root)
        print(json.dumps(report, ensure_ascii=False, indent=2))
        if not report["ready"]:
            raise ValueError("Expected assets are missing. JSON generation is not asset generation.")
    else:
        plan = gaussian_plan(read(args.jobs), read(args.config), args.project_root)
        write(Path(args.output) / "plan.json", plan)
        if args.execute:
            execute(plan, args.project_root, args.output)
        else:
            print("Plan saved. Use --execute to run it; ML has not been started.")


if __name__ == "__main__":
    try:
        main()
    except (ValueError, KeyError, OSError, RuntimeError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        sys.exit(1)
