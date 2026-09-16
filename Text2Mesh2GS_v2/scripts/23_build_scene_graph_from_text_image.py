#!/usr/bin/env python3
"""Recognize and verify individual room objects using a local VLM."""
import argparse
from pathlib import Path
from local_scene import image_info, load_config, bridge
from scene_review import recognize


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/scene_pipeline.yaml')
    p.add_argument('--scene_id', required=True)
    p.add_argument('--image_path', required=True)
    p.add_argument('--output')
    p.add_argument('--anchors', help='JSON mapping fixed-object IDs to verified room-local transforms')
    p.add_argument('--review-decisions', help='JSON mapping detected IDs to human-reviewed decision records')
    args = p.parse_args()
    if not args.scene_id or any(c in args.scene_id for c in '/\\') or args.scene_id in {'.', '..'}:
        raise ValueError('Invalid scene_id')
    config = load_config(args.config)
    reference, image = image_info(args.image_path)
    out = Path(args.output or f'outputs/scene_graphs/{args.scene_id}/scene_graph.json')
    scene = recognize(config['local_llm'], reference, image, args.scene_id,
                      config['scene_graph']['default_room_size'], out,
                      bridge.read(args.anchors) if args.anchors else {},
                      bridge.read(args.review_decisions) if args.review_decisions else {})
    bridge.write(out, scene)
    pending = scene['generation']['unresolved']
    print(f'Saved {out}; review status: {scene["generation"]["review_status"]}; unresolved: {len(pending)}')
    if pending:
        print('Review generation.unresolved and the .review audit directory before stage 34.')
        raise SystemExit(2)


if __name__ == '__main__':
    main()
