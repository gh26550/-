"""Staged local visual review and source-scoped constraint generation."""
import base64
import copy
import hashlib
import io
import json
from pathlib import Path

from PIL import Image
from local_scene import (TEXT, ITEM, CONSTRAINT, arr, obj_schema, generate,
                         check_inventory, build_scene, constraint_schema,
                         check_constraints, RELATION_HELP, bridge)


def cached_generate(settings, prompt, schema, images, validate, audit_path):
    """Reuse only successful same-input results. Failed/uncertain results are retried."""
    key = hashlib.sha256(json.dumps([settings, prompt, schema, images], sort_keys=True).encode()).hexdigest()
    cache = audit_path.with_suffix('.cache.json')
    if cache.exists():
        saved = bridge.read(cache)
        if saved.get('key') == key:
            import jsonschema
            jsonschema.validate(saved['result'], schema)
            validate(saved['result'])
            return saved['result']
    result = generate(settings, prompt, schema, images, validate, audit_path)
    if result.get('status') != 'needs_review':
        bridge.write(cache, {'key': key, 'result': result})
    return result

DETECTION = obj_schema({k: ITEM['properties'][k] for k in ('id', 'name', 'category', 'bbox')})
DETECTIONS = obj_schema({'summary': TEXT, 'uncertainties': arr(TEXT),
                         'objects': arr(DETECTION, maxItems=64)})
DIMENSIONS = obj_schema({k: {'type': 'number', 'exclusiveMinimum': 0}
                         for k in ('width', 'height', 'depth')})


def fingerprint(scene):
    return hashlib.sha256(json.dumps(scene, sort_keys=True, ensure_ascii=False).encode()).hexdigest()


def crop_image(path, box):
    """Include surrounding support context; retain the full image in every request."""
    with Image.open(path) as im:
        x1, y1, x2, y2 = box
        dx, dy = max(32, (x2-x1)*.35), max(32, (y2-y1)*.35)
        bounds = (max(0, int(x1-dx)), max(0, int(y1-dy)),
                  min(im.width, int(x2+dx+1)), min(im.height, int(y2+dy+1)))
        if bounds[2] <= bounds[0] or bounds[3] <= bounds[1]:
            raise ValueError('Invalid crop box')
        stream = io.BytesIO()
        im.crop(bounds).convert('RGB').save(stream, format='PNG')
        return base64.b64encode(stream.getvalue()).decode()


def check_detections(data, reference):
    ids = set()
    for row in data['objects']:
        if row['id'] in ids or row['id'] in {'floor_01', 'ceiling_01', 'wall_left_01', 'wall_right_01', 'wall_back_01'}:
            raise ValueError('Duplicate/reserved ID: ' + row['id'])
        ids.add(row['id'])
        x1, y1, x2, y2 = row['bbox']
        if not (0 <= x1 < x2 <= reference['width'] and 0 <= y1 < y2 <= reference['height']):
            raise ValueError('Pixel bbox outside image: ' + row['id'])


def overlap_candidates(item, rows):
    result = []
    a = item['bbox']
    for row in rows:
        if row['id'] == item['id']:
            continue
        b = row['bbox']
        intersection = max(0, min(a[2], b[2])-max(a[0], b[0])) * max(0, min(a[3], b[3])-max(a[1], b[1]))
        smaller = min((a[2]-a[0])*(a[3]-a[1]), (b[2]-b[0])*(b[3]-b[1]))
        if smaller and intersection / smaller > .35:
            result.append(row['id'])
    return result


def recognize(settings, reference, image, scene_id, size, out, anchors=None, review_decisions=None):
    anchors = anchors or {}
    review_decisions = review_decisions or {}
    for oid, transform in anchors.items():
        for key in ('position', 'rotation_euler', 'scale'):
            bridge.vector(transform[key], oid + '.' + key, key == 'scale')
    audit = out.parent / (out.stem + '.review')
    prompt = f'''Detect visible distinct room objects, including supports (tables, cabinets,
shelves), windows and visible outside views. Return ONLY identity and pixel xyxy boxes.
Image size {reference['width']}x{reference['height']}. Windows use category window_frame. Do not fill a quota. Separate foreground
and rear tables if visibly distinct. Plants must be independent plants/pots, not individual
branches, leaves or outdoor trees. Cushions/books belong to their furniture. Include fixed
shelves as candidates. Exclude supplied floor/wall/ceiling anchors. Do not infer hidden objects.'''
    data = cached_generate(settings, prompt, DETECTIONS, image,
                    lambda d: check_detections(d, reference), audit / 'detection.json')
    data = cached_generate(settings, prompt + '\nCheck omissions and duplicate identities against the whole image. '
                    'Preserve correct IDs. Return corrected complete candidates:\n' + json.dumps(data),
                    DETECTIONS, image, lambda d: check_detections(d, reference), audit / 'detection_review.json')
    rows = data['objects']
    ids = [r['id'] for r in rows]
    if set(review_decisions) - set(ids) or set(anchors) - set(ids):
        raise ValueError('Manual review/anchor IDs must match current detections')
    schema = obj_schema({
        'status': {'enum': ['confirmed', 'part_or_duplicate', 'outdoor_background', 'needs_review']},
        'related_id': {'enum': ids + ['unknown']},
        'reason': TEXT, 'representation': {'enum': ['mesh', 'gaussian', 'procedural']},
        'support_id': {'enum': ids + ['floor_01', 'unknown', 'not_applicable']},
        'mounting': {'enum': ['supported', 'wall_mounted', 'window', 'background', 'unknown']},
        'estimated_dimensions_m': DIMENSIONS, 'dimensions_plausible': {'type': 'boolean'},
        'description': TEXT, 'asset_prompt': TEXT})
    decisions, accepted, unresolved = {}, [], []
    for index, row in enumerate(rows):
        print(f"Review {index+1}/{len(rows)}: {row['id']}", flush=True)
        def validate(value):
            if value['status'] == 'part_or_duplicate' and value['related_id'] in {'unknown', row['id']}:
                raise ValueError('Identify a different parent/duplicate ID or mark needs_review')
            if value['support_id'] == row['id']:
                raise ValueError('Object cannot support itself')
            if value['representation'] == 'procedural' and row['category'] != 'window_frame':
                raise ValueError('Only category window_frame is procedural; shelves are mesh')
            if (value['representation'] == 'gaussian') != (value['mounting'] == 'background'):
                raise ValueError('Gaussian backgrounds must use mounting=background; furniture must not')
            if value['representation'] == 'procedural' and value['mounting'] != 'window':
                raise ValueError('Procedural window must use mounting=window')
        try:
            prompt = ('Image 1 is the whole room; image 2 is a context crop. '
                'Verify ONLY this candidate. Compare overlapping candidates, but overlap alone does not mean duplicate. '
                'Indoor independent objects and outside-view plates may be confirmed; outdoor trees mistakenly '
                'detected as indoor meshes are outdoor_background. Wall shelves are wall_mounted. '
                'Use unknown support if occluded; do not invent. Dimensions: width X, height Y, depth Z in metres; '
                'check thin rugs and table height. State visible evidence. Give isolated English asset prompt.\n' +
                json.dumps({'candidate': row, 'overlap_candidates': overlap_candidates(row, rows), 'inventory': rows}))
            if row['id'] in review_decisions:
                import jsonschema
                value = review_decisions[row['id']]
                jsonschema.validate(value, schema)
                validate(value)
                bridge.write(audit / (row['id'] + '.manual.json'), {'method': 'human_review', 'result': value})
            else:
                value = cached_generate(settings, prompt, schema, [image, crop_image(reference['path'], row['bbox'])],
                                        validate, audit / (row['id'] + '.json'))
        except ValueError as exc:
            unresolved.append({'id': row['id'], 'reason': str(exc)})
            continue
        decisions[row['id']] = value
        if value['status'] == 'part_or_duplicate' or value['status'] == 'outdoor_background':
            continue
        fixed = value['mounting'] in {'wall_mounted', 'window'}
        if (value['status'] != 'confirmed' or not value['dimensions_plausible'] or
            value['mounting'] == 'unknown' or (fixed and row['id'] not in anchors) or
            (not fixed and value['representation'] == 'mesh' and value['support_id'] in {'unknown', 'not_applicable'})):
            reason = ('Verified fixed transform required via --anchors. ' if fixed and row['id'] not in anchors else '') + value['reason']
            unresolved.append({'id': row['id'], 'reason': reason, 'review': value})
            continue
        dims = value['estimated_dimensions_m']
        accepted.append({**row, 'representation': value['representation'],
            'movable': value['representation'] == 'mesh', 'description': value['description'],
            'asset_prompt': value['asset_prompt'], 'dimensions_m': [dims[k] for k in ('width', 'height', 'depth')],
            'confidence': 0.5, 'evidence': value['reason']})
    accepted_ids = {r['id'] for r in accepted} | {'floor_01'}
    for oid, value in decisions.items():
        if value['status'] == 'part_or_duplicate' and value['related_id'] not in accepted_ids:
            unresolved.append({'id': oid, 'reason': 'Parent/duplicate target is not confirmed'})
    for row in accepted:
        value = decisions[row['id']]
        if value['representation'] == 'mesh' and value['mounting'] != 'wall_mounted' and value['support_id'] not in accepted_ids:
            unresolved.append({'id': row['id'], 'reason': 'Support is not confirmed: ' + value['support_id']})
    if len(rows) == 64:
        unresolved.append({'id': 'inventory', 'reason': 'Detection limit reached; review omissions'})
    inventory = {'summary': data['summary'], 'uncertainties': data['uncertainties'], 'objects': accepted}
    scene = build_scene(inventory, scene_id, size, reference, settings)
    for obj in scene['objects']:
        if obj['id'] in decisions:
            value = decisions[obj['id']]
            obj.update(support_id=value['support_id'], placement_state='confirmed',
                       estimated_dimensions_m=value['estimated_dimensions_m'])
            obj['visual_evidence']['confidence_kind'] = 'unscored_placeholder'
            if value['mounting'] in {'wall_mounted', 'window'}:
                obj['transform'] = anchors[obj['id']]
                obj['properties']['movable'] = False
                obj['properties']['interaction_required'] = False
                obj['interaction_required'] = False
                obj['placement_state'] = 'fixed_verified'
    scene['generation'].update(review_pipeline='per_object_v1', review_status='needs_review' if unresolved else 'complete',
                               unresolved=unresolved, detections=rows, decisions=decisions)
    return bridge.normalise(scene)


def generate_constraints(settings, scene, actual, image, out):
    records, constraints = [], []
    # Invalidate any older final result before inference or a possible connection failure.
    base = {'schema_version': bridge.VERSION, 'scene_id': scene['scene_id'], 'status': 'incomplete',
            'scene_fingerprint': fingerprint(scene), 'reference_image': actual,
            'method': 'local_vision_constraints', 'model': settings['model']}
    bridge.write(out, {**base, 'constraints': [], 'unresolved': ['generation in progress']})
    sources = [o for o in scene['objects'] if o['properties']['movable'] or o['representation'] == 'gaussian']
    inventory = [{k: o.get(k) for k in ('id', 'category', 'support_id', 'visual_evidence')} for o in scene['objects']]
    audit = out.parent / (out.stem + '.objects')
    for index, source in enumerate(sources):
        oid = source['id']
        print(f'Constraints {index+1}/{len(sources)}: {oid}', flush=True)
        schema = constraint_schema(scene)
        schema['properties'].update(status={'enum': ['resolved', 'needs_review']}, reason=TEXT)
        schema['required'] += ['status', 'reason']
        # Use exactly the source's representation-specific relations, not another branch.
        schema['properties']['constraints']['items'] = copy.deepcopy(CONSTRAINT)
        props = schema['properties']['constraints']['items']['properties']
        props['source'] = {'enum': [oid]}
        background = source['representation'] == 'gaussian'
        props['relation'] = {'enum': sorted({'behind', 'center aligned with', 'aligned with', 'parallel to', 'cover', 'visible through'} if background else bridge.FURNITURE)}
        props['target'] = {'enum': [o['id'] for o in scene['objects'] if o['id'] != oid and
                                    (not background or o['category'] in {'wall', 'window_frame'})] + ([] if background else ['room_center'])}
        props['basis'] = {'enum': ['observed', 'design_intent']}
        schema['properties']['constraints']['items']['required'].append('basis')
        images = [image]
        box = source.get('visual_evidence', {}).get('bbox_pixels')
        if box:
            images.append(crop_image(actual['path'], box))
        support_obj = next((o for o in scene['objects'] if o['id'] == source.get('support_id')), None)
        support_box = (support_obj or {}).get('visual_evidence', {}).get('bbox_pixels')
        if support_box:
            images.append(crop_image(actual['path'], support_box))
        prompt = RELATION_HELP + '\nOnly source ' + oid + '''. Image 1 is whole room, image 2 (if present) is its context crop.
Image 3, if present, is the verified support context crop.
Return needs_review with empty constraints if placement cannot be justified. Do not force a relation.
Distinguish observation from inferred design_intent using basis. Honor verified support_id with an on relation.
Check whether a claimed wall contact is actually visible; image-left is not wall contact.
Return resolved only after checking evidence against image. Existing accepted constraints must remain consistent.\n''' + json.dumps({'objects': inventory, 'accepted_constraints': constraints})
        def validate(value):
            if value['status'] == 'needs_review':
                if value['constraints']:
                    raise ValueError('Unresolved source must have no accepted constraints')
                return
            if not value['constraints']:
                raise ValueError('Resolved source requires a placement relation')
            support = source.get('support_id')
            if source['properties']['movable'] and support and not any(c['relation'] == 'on' and c['target'] == support for c in value['constraints']):
                raise ValueError('Missing verified on support: ' + support)
            check_constraints({'constraints': constraints + value['constraints']}, scene, require_coverage=False)
        try:
            value = cached_generate(settings, prompt, schema, images, validate, audit / (oid + '.json'))
            if value['status'] == 'resolved':
                value = cached_generate(settings, prompt + '\nIndependently review this proposed result against the images; '
                    'correct unsupported relations or mark needs_review:\n' + json.dumps(value), schema, images,
                    validate, audit / (oid + '.review.json'))
            records.append({'id': oid, **value})
            constraints.extend(value['constraints'])
        except ValueError as exc:
            records.append({'id': oid, 'status': 'needs_review', 'reason': str(exc), 'constraints': []})
        bridge.write(out.with_suffix('.draft.json'), {**base, 'records': records, 'constraints': constraints})
    unresolved = [r for r in records if r['status'] != 'resolved']
    status = 'incomplete' if unresolved else 'complete'
    result = {'schema_version': bridge.VERSION, 'scene_id': scene['scene_id'], 'status': status,
              'scene_fingerprint': fingerprint(scene), 'reference_image': actual, 'method': 'local_vision_constraints',
              'model': settings['model'], 'constraints': constraints, 'records': records, 'unresolved': unresolved}
    if not unresolved:
        check_constraints(result, scene)
    bridge.write(out, result)
    return result
