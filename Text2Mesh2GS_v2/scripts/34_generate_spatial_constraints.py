#!/usr/bin/env python3
"""Generate image-grounded constraints with a local LLM and semantic validation."""
import argparse
import json
from pathlib import Path
from local_scene import constraint_schema, RELATION_HELP, check_constraints, generate, image_info, load_config, bridge


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", default="configs/scene_pipeline.yaml")
    p.add_argument("--scene_graph", required=True)
    p.add_argument("--image_path", help="Relocated image, must match original SHA256")
    p.add_argument("--output", required=True)
    args = p.parse_args()
    config = load_config(args.config)
    scene = bridge.read(args.scene_graph)
    bridge.validate(scene)
    reference = scene["reference_image"]
    actual, image = image_info(args.image_path or reference["path"])
    if actual["sha256"] != reference["sha256"]:
        raise ValueError("Image differs from the image used for furniture recognition")
    inventory = [{k: o.get(k) for k in ("id", "name", "category", "representation", "properties", "visual_evidence")} for o in scene["objects"]]
    prompt = RELATION_HELP + "\nGenerate constraints ONLY for the existing inventory IDs below. Never invent missing furniture IDs even if visible. Fixed windows/walls/floors are targets, never sources. Use the room-size setting as scale context, not inferred measurement.\n" + json.dumps({"room": scene["room"], "objects": inventory}, ensure_ascii=False)
    out = Path(args.output)
    data = generate(config["local_llm"], prompt, constraint_schema(scene), image,
                    lambda value: check_constraints(value, scene), out.with_suffix(".llm_audit.json"))
    bridge.write(out, {"schema_version": bridge.VERSION, "scene_id": scene["scene_id"],
                       "method": "local_vision_constraints", "model": config["local_llm"]["model"],
                       "reference_image": actual, **data})
    print(f"Saved {len(data['constraints'])} validated LLM constraints to {out}")


if __name__ == "__main__":
    main()
