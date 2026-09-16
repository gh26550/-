#!/usr/bin/env python3
"""Generate and visually review constraints separately for each verified object."""
import argparse
import datetime
from pathlib import Path
from local_scene import image_info, load_config, bridge
from scene_review import generate_constraints, repair_scene_supports, check_placement_contract


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--config', default='configs/scene_pipeline.yaml')
    p.add_argument('--scene_graph', required=True)
    p.add_argument('--image_path', help='Relocated image; SHA256 must match')
    p.add_argument('--output', required=True)
    p.add_argument('--estimate-fixed', action='store_true', help='Authorize image-based approximate fixed window positions')
    args = p.parse_args()
    config = load_config(args.config)
    scene = bridge.read(args.scene_graph)
    bridge.validate(scene)
    generation = scene.get('generation', {})
    if generation.get('review_pipeline') != 'per_object_v1':
        raise ValueError('Run updated stage 23 and resolve generation.unresolved before stage 34. Old inventories must be reviewed again.')
    actual, image = image_info(args.image_path or scene['reference_image']['path'])
    if actual['sha256'] != scene['reference_image']['sha256']:
        raise ValueError('Image differs from recognition image')
    repaired = repair_scene_supports(config['local_llm'], scene, actual, image,
                                    Path(args.output).parent / 'support_review', args.estimate_fixed)
    if repaired != scene:
        backup = Path(args.scene_graph).with_suffix('.before_support_review.' + datetime.datetime.now().strftime('%Y%m%d_%H%M%S_%f') + '.json')
        bridge.write(backup, scene)
        bridge.write(args.scene_graph, repaired)
        scene = repaired
        print('Updated support review; original saved to ' + str(backup))
    if scene['generation']['unresolved']:
        bridge.write(args.output, {'schema_version': bridge.VERSION, 'scene_id': scene['scene_id'],
                                  'status': 'incomplete', 'constraints': [], 'unresolved': scene['generation']['unresolved']})
        print('Support/fixed-anchor review incomplete: ' + str(scene['generation']['unresolved']))
        raise SystemExit(2)
    check_placement_contract(scene)
    result = generate_constraints(config['local_llm'], scene, actual, image, Path(args.output))
    print(f'Saved {args.output}: {result["status"]}, {len(result["unresolved"])} unresolved objects')
    if result['status'] != 'complete':
        raise SystemExit(2)


if __name__ == '__main__':
    main()
