#!/usr/bin/env python3
"""
Step 2: 参照画像 + テキストから scene_graph.json を作る。

現段階では、VLMを使わない rule-based 版です。
参照画像のパスは scene_graph に記録しますが、画像内容の自動認識は行いません。
まずは後段の Mesh / Gaussian 分割とUnity配置まで流すための初期 scene_graph を作ります。

入力:
  configs/scene_pipeline.yaml
  data/scene_prompts.csv
  outputs/scene_images/<scene_id>/candidate_00.png など

出力:
  outputs/scene_graphs/<scene_id>/scene_graph.json

実行例:
  python scripts/23_build_scene_graph_from_text_image.py \
    --config configs/scene_pipeline.yaml \
    --scene_id room_001 \
    --image_path outputs/scene_images/room_001/candidate_02.png
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml


def load_yaml(path: Path) -> Dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def resolve_project_path(config_path: Path, maybe_relative_path: str) -> Path:
    """設定ファイルの場所ではなく、実行中のプロジェクトルート基準で相対パスを解決する。"""
    p = Path(maybe_relative_path)
    if p.is_absolute():
        return p
    return Path.cwd() / p


def clean_csv_value(value: Optional[str]) -> str:
    """CSVセルの前後空白・改行・余分な外側引用符を整理する。"""
    if value is None:
        return ""
    s = value.strip()
    while len(s) >= 2 and s.startswith('"') and s.endswith('"'):
        s = s[1:-1].strip()
    return s


def read_scene_prompts(csv_path: Path) -> List[Dict[str, str]]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Prompt CSV not found: {csv_path}")

    encodings = ["utf-8-sig", "utf-8", "cp932", "shift_jis"]
    last_error: Optional[UnicodeDecodeError] = None

    for enc in encodings:
        try:
            rows: List[Dict[str, str]] = []
            with csv_path.open("r", encoding=enc, newline="") as f:
                reader = csv.DictReader(f)
                required = {"scene_id", "scene_name", "prompt"}
                missing = required - set(reader.fieldnames or [])
                if missing:
                    raise ValueError(f"Prompt CSV is missing columns: {sorted(missing)}")

                for row in reader:
                    scene_id = clean_csv_value(row.get("scene_id"))
                    scene_name = clean_csv_value(row.get("scene_name"))
                    prompt = clean_csv_value(row.get("prompt"))
                    if not scene_id or not prompt:
                        continue
                    rows.append({
                        "scene_id": scene_id,
                        "scene_name": scene_name,
                        "prompt": prompt,
                    })

            if not rows:
                raise ValueError(f"No valid scene prompts found in: {csv_path}")

            print(f"Loaded prompt CSV with encoding: {enc}")
            return rows

        except UnicodeDecodeError as e:
            last_error = e
            continue

    raise UnicodeDecodeError(
        "utf-8/cp932/shift_jis",
        b"",
        0,
        1,
        f"Could not decode {csv_path}. Last error: {last_error}",
    )


def find_scene(rows: List[Dict[str, str]], scene_id: str) -> Dict[str, str]:
    for row in rows:
        if row["scene_id"] == scene_id:
            return row
    raise ValueError(f"scene_id not found in prompt CSV: {scene_id}")


def has_any(text: str, keywords: List[str]) -> bool:
    lower = text.lower()
    return any(k.lower() in lower for k in keywords)


def transform(position, rotation=(0.0, 0.0, 0.0), scale=(1.0, 1.0, 1.0)) -> Dict[str, List[float]]:
    return {
        "position": [float(v) for v in position],
        "rotation_euler": [float(v) for v in rotation],
        "scale": [float(v) for v in scale],
    }


def obj(
    object_id: str,
    name: str,
    category: str,
    representation: str,
    role: str,
    interaction_required: bool,
    background_only: bool,
    position,
    rotation=(0.0, 0.0, 0.0),
    scale=(1.0, 1.0, 1.0),
    object_prompt: Optional[str] = None,
) -> Dict[str, Any]:
    return {
        "id": object_id,
        "name": name,
        "category": category,
        "room_role": role,
        "representation": representation,
        "interaction_required": interaction_required,
        "background_only": background_only,
        "transform": transform(position, rotation, scale),
        "asset_prompt": object_prompt or name,
    }


def build_rule_based_scene_graph(
    scene: Dict[str, str],
    config: Dict[str, Any],
    reference_image: Optional[Path],
) -> Dict[str, Any]:
    scene_id = scene["scene_id"]
    scene_name = scene.get("scene_name", "")
    prompt = scene["prompt"]
    text = f"{scene_name} {prompt}"

    room_cfg = config.get("scene_graph", {})
    default_room_size = room_cfg.get("default_room_size", [6.0, 4.0, 6.0])

    objects: List[Dict[str, Any]] = []

    # 部屋構造は、Unityで空間を成立させるためMeshとして保持する。
    objects.extend([
        obj(
            "floor_01", "light wooden floor", "floor", "mesh",
            "walkable room base", False, False,
            position=(0.0, 0.0, 0.0), scale=(6.0, 1.0, 6.0),
            object_prompt="simple light nordic wooden floor plane"
        ),
        obj(
            "ceiling_01", "white ceiling", "ceiling", "mesh",
            "room boundary", False, False,
            position=(0.0, 4.0, 0.0), scale=(6.0, 1.0, 6.0),
            object_prompt="simple white flat ceiling plane"
        ),
        obj(
            "wall_back_01", "back white wall", "wall", "mesh",
            "room boundary with window", False, False,
            position=(0.0, 2.0, 3.0), scale=(6.0, 4.0, 1.0),
            object_prompt="simple white interior wall with large window opening"
        ),
        obj(
            "wall_left_01", "left white wall", "wall", "mesh",
            "room boundary", False, False,
            position=(-3.0, 2.0, 0.0), rotation=(0.0, 90.0, 0.0), scale=(6.0, 4.0, 1.0),
            object_prompt="simple white interior side wall"
        ),
        obj(
            "wall_right_01", "right white wall", "wall", "mesh",
            "room boundary", False, False,
            position=(3.0, 2.0, 0.0), rotation=(0.0, 90.0, 0.0), scale=(6.0, 4.0, 1.0),
            object_prompt="simple white interior side wall"
        ),
    ])

    if has_any(text, ["window", "窓"]):
        objects.append(
            obj(
                "window_frame_01", "large back window frame", "window_frame", "mesh",
                "physical boundary between room and outside view", False, False,
                position=(0.0, 2.0, 2.95), rotation=(0.0, 0.0, 0.0), scale=(3.2, 2.2, 0.1),
                object_prompt="large white nordic window frame, simple rectangular frame"
            )
        )

    # リビングの参照画像ではソファが出やすいため、living_room系では初期Meshとして入れる。
    if has_any(text, ["living room", "living_room", "リビング"]):
        objects.append(
            obj(
                "sofa_01", "light gray sofa", "sofa", "mesh",
                "main seating furniture", True, False,
                position=(1.0, 0.0, 1.2), rotation=(0.0, 180.0, 0.0), scale=(1.7, 1.0, 1.0),
                object_prompt="light gray nordic sofa, simple clean design, front view, isolated object"
            )
        )
        objects.append(
            obj(
                "rug_01", "light gray rug", "rug", "mesh",
                "floor decoration and spatial grouping", False, False,
                position=(0.4, 0.02, 0.6), rotation=(0.0, 0.0, 0.0), scale=(2.5, 1.0, 2.0),
                object_prompt="simple light gray rectangular rug, top view, isolated object"
            )
        )

    if has_any(text, ["chair", "椅子", "チェア"]):
        objects.append(
            obj(
                "chair_01", "wooden chair", "chair", "mesh",
                "sittable furniture", True, False,
                position=(-1.7, 0.0, 0.6), rotation=(0.0, 30.0, 0.0), scale=(1.0, 1.0, 1.0),
                object_prompt="wooden nordic chair, front view, isolated object, white background"
            )
        )

    if has_any(text, ["round table", "table", "丸テーブル", "テーブル"]):
        objects.append(
            obj(
                "table_01", "round wooden coffee table", "table", "mesh",
                "support surface for objects", True, False,
                position=(0.0, 0.0, 0.4), rotation=(0.0, 0.0, 0.0), scale=(1.2, 1.0, 1.2),
                object_prompt="round wooden nordic coffee table, front view, isolated object, white background"
            )
        )

    if has_any(text, ["bookshelf", "book shelf", "本棚", "書棚"]):
        objects.append(
            obj(
                "bookshelf_01", "wooden bookshelf", "bookshelf", "mesh",
                "storage furniture and physical obstacle", True, False,
                position=(-2.4, 0.0, 2.2), rotation=(0.0, 90.0, 0.0), scale=(1.0, 1.4, 0.7),
                object_prompt="wooden nordic bookshelf with books, front view, isolated object, white background"
            )
        )

    if has_any(text, ["vase", "花瓶"]):
        objects.append(
            obj(
                "vase_01", "white ceramic vase", "vase", "mesh",
                "small interactable decorative object", True, False,
                position=(-0.2, 0.55, 0.35), rotation=(0.0, 0.0, 0.0), scale=(0.35, 0.35, 0.35),
                object_prompt="white ceramic vase, front view, isolated object, white background"
            )
        )

    if has_any(text, ["plant", "potted plant", "観葉植物", "植物"]):
        objects.append(
            obj(
                "plant_01", "large potted plant", "plant", "mesh",
                "near-room decorative object that may be inspected or avoided", True, False,
                position=(2.2, 0.0, 1.9), rotation=(0.0, -30.0, 0.0), scale=(1.0, 1.0, 1.0),
                object_prompt="large indoor potted plant, front view, isolated object, white background"
            )
        )

    # 窓外・遠景・空気感はGaussianとして扱う。
    if has_any(text, ["outside", "street", "tree", "building", "window", "窓の外", "街路樹", "建物", "外"]):
        objects.append(
            obj(
                "outside_view_01", "outside view with street trees and buildings", "outside_view", "gaussian",
                "distant visual background seen through the window", False, True,
                position=(0.0, 2.0, 3.4), rotation=(0.0, 180.0, 0.0), scale=(3.6, 2.4, 1.0),
                object_prompt="soft distant outside view through a large window, street trees, low buildings, bright daylight"
            )
        )

    rw, rh, rd = [float(v) for v in default_room_size]
    if min(rw, rh, rd) <= 0:
        raise ValueError("Room dimensions must be positive")
    for o in objects:
        t = o["transform"]
        if o["category"] in {"floor", "ceiling"}:
            t["position"] = [0.0, rh if o["category"] == "ceiling" else 0.0, 0.0]
            t["scale"] = [rw, 1.0, rd]
        elif o["id"] == "wall_back_01":
            t["position"] = [0.0, rh / 2, rd / 2]
            t["scale"] = [rw, rh, 1.0]
        elif o["id"] in {"wall_left_01", "wall_right_01"}:
            t["position"] = [(-1 if "left" in o["id"] else 1) * rw / 2, rh / 2, 0.0]
            t["scale"] = [rd, rh, 1.0]

    scene_graph: Dict[str, Any] = {
        "scene_id": scene_id,
        "scene_name": scene_name,
        "source_prompt": prompt,
        "reference_image": str(reference_image.as_posix()) if reference_image else None,
        "method": "rule_based_v0",
        "note": "This version records the reference image path but does not perform visual recognition. Replace this step with a VLM-based parser later if needed.",
        "room": {
            "size": [float(v) for v in default_room_size],
            "coordinate_system": "Unity-like: x=right, y=up, z=forward/depth",
        },
        "representation_policy": {
            "mesh": "interactable objects, physical boundaries, walkable/support surfaces",
            "gaussian": "background-only, distant, visual-atmosphere elements",
            "hybrid": "disabled",
        },
        "objects": objects,
    }
    return scene_graph


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build initial scene_graph.json from scene prompt and reference image path.")
    parser.add_argument("--config", type=str, required=True, help="Path to configs/scene_pipeline.yaml")
    parser.add_argument("--scene_id", type=str, required=True, help="Target scene_id, e.g., room_001")
    parser.add_argument("--image_path", type=str, default=None, help="Reference scene image path. If omitted, candidate_00.png is used.")
    parser.add_argument("--output", type=str, default=None, help="Output scene_graph.json path. If omitted, outputs/scene_graphs/<scene_id>/scene_graph.json")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    config = load_yaml(config_path)

    prompt_csv = resolve_project_path(config_path, config["scene_input"]["prompt_csv"])
    rows = read_scene_prompts(prompt_csv)
    scene = find_scene(rows, args.scene_id)

    output_root = resolve_project_path(config_path, config.get("project", {}).get("output_root", "outputs"))

    if args.image_path:
        reference_image = Path(args.image_path)
        if not reference_image.is_absolute():
            reference_image = Path.cwd() / reference_image
    else:
        reference_image = output_root / "scene_images" / args.scene_id / "candidate_00.png"

    if not reference_image.exists():
        print(f"Warning: reference image not found: {reference_image}")
        print("scene_graph will still be created with reference_image path recorded.")

    if args.output:
        output_path = Path(args.output)
        if not output_path.is_absolute():
            output_path = Path.cwd() / output_path
    else:
        output_path = output_root / "scene_graphs" / args.scene_id / "scene_graph.json"

    scene_graph = build_rule_based_scene_graph(scene, config, reference_image)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(scene_graph, f, ensure_ascii=False, indent=2)

    mesh_count = sum(1 for o in scene_graph["objects"] if o["representation"] == "mesh")
    gaussian_count = sum(1 for o in scene_graph["objects"] if o["representation"] == "gaussian")

    print(f"Saved scene_graph: {output_path}")
    print(f"Objects: total={len(scene_graph['objects'])}, mesh={mesh_count}, gaussian={gaussian_count}")


if __name__ == "__main__":
    main()
