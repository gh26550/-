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
            try:
                jsonschema.validate(saved['result'], schema)
                validate(saved['result'])
                if saved['result'].get('support_id') != 'unknown' and saved['result'].get('status') != 'needs_review':
                    return saved['result']
            except (ValueError, KeyError, jsonschema.ValidationError):
                pass  # New validation rules must invalidate previously accepted mistakes.
    result = generate(settings, prompt, schema, images, validate, audit_path)
    if result.get('status') != 'needs_review' and result.get('support_id') != 'unknown':
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


def possible_supports(row, rows):
    """Projection overlap is necessary, not sufficient, for visible object contact."""
    a = row['bbox']
    margin = max(8, (a[2] - a[0]) * .05)
    return [r['id'] for r in rows if r['id'] != row['id'] and
            r['category'] != 'window_frame' and
            min(a[2], r['bbox'][2]) + margin >= max(a[0], r['bbox'][0])]


def verify_support(settings, row, rows, reference, image, audit):
    candidates = possible_supports(row, rows)
    schema = obj_schema({'evidence': TEXT, 'support_id': TEXT})
    prompt = ('Determine ONLY the physical surface directly underneath the specified object. '
              'Support means carries its weight by contact, NOT nearby, behind, facing, or belonging to the same room. '
              'floor_01 means the configured room floor, including a floor covered by a rug not separately modeled. '
              'If the object stands on that floor/rug, answer floor_01; you do not need a named floor patch or exact contact point. '
              'Do not choose the first ID. If contact is unclear choose unknown. '
              'The whole image is followed by a context crop of the source. Explain the contact, not merely object identity.\n' +
              json.dumps({'source': row, 'allowed_support_ids': ['floor_01'] + candidates + ['unknown'],
                          'candidate_supports': [r for r in rows if r['id'] in candidates], 'floor_01': 'room floor'}))
    def validate(value):
        if value['support_id'] not in ['floor_01', 'unknown'] + candidates:
            raise ValueError('Use one of the allowed support IDs, or unknown')
    return cached_generate(settings, prompt, schema, [image, crop_image(reference['path'], row['bbox'])],
                           validate, audit)


def check_placement_contract(scene):
    """Prevent poisoned stage-23 state from becoming mandatory solver constraints."""
    for obj in scene['objects']:
        if obj['category'] == 'window_frame' and obj['properties']['movable']:
            raise ValueError('Window cannot be movable furniture: ' + obj['id'] + '; rerun support review')
        if obj['properties']['movable'] and obj.get('support_validation_version') != 3:
            raise ValueError('Support needs independent verification: ' + obj['id'] + '; rerun support review')


def check_relation_geometry(constraints, scene):
    for oid in {c['source'] for c in constraints}:
        walls = {c['target'] for c in constraints if c['source'] == oid and c['relation'] == 'against'}
        if {'wall_left_01', 'wall_right_01'} <= walls:
            raise ValueError('Cannot require contact with opposite room walls: ' + oid)
        for target in walls:
            if next(o for o in scene['objects'] if o['id'] == target)['category'] != 'wall':
                raise ValueError('against requires a wall target: ' + target)


def estimate_window(settings, obj, scene, reference, image, previous, audit):
    fraction = {'type': 'number', 'minimum': .03, 'maximum': .97}
    schema = obj_schema({'wall': {'enum': ['wall_back_01', 'wall_left_01', 'wall_right_01']},
                         'left_fraction': fraction, 'right_fraction': fraction,
                         'bottom_fraction': fraction, 'top_fraction': fraction, 'evidence': TEXT})
    w, h, d = scene['room']['size']
    def convert(value):
        span = w if value['wall'] == 'wall_back_01' else d
        if value['right_fraction'] <= value['left_fraction'] or value['top_fraction'] <= value['bottom_fraction']:
            raise ValueError('Window fractions require left < right and bottom < top')
        return {'wall': value['wall'], 'horizontal_m': ((value['left_fraction']+value['right_fraction'])/2-.5)*span,
                'center_height_m': (value['bottom_fraction']+value['top_fraction'])/2*h,
                'width_m': (value['right_fraction']-value['left_fraction'])*span,
                'height_m': (value['top_fraction']-value['bottom_fraction'])*h, 'evidence': value['evidence']}
    def validate(raw):
        value = convert(raw)
    value = cached_generate(settings,
        'Propose an approximate fixed WINDOW OPENING for Unity reconstruction, NOT measured coordinates. '
        'Return a rectangle as FRACTIONS of the selected wall (0 to 1), NOT metres or image pixels. '
        '0=wall left/bottom, 1=wall right/top. Estimate only the specified window, excluding curtains/plants. '
        'Back wall horizontal coordinate is room X; side wall horizontal coordinate is room Z. '
        'Image left/right does not necessarily mean side wall. Respect the image and existing proposals.\n' +
        json.dumps({'window': obj, 'room_size': scene['room']['size'], 'previous_windows': previous}),
        schema, [image, crop_image(reference['path'], obj['visual_evidence']['bbox_pixels'])], validate, audit)
    value = convert(value)
    x, y = value['horizontal_m'], value['center_height_m']
    position, rotation = ([x, y, d/2], [0, 0, 0]) if value['wall'] == 'wall_back_01' else (
        ([-w/2, y, x], [0, 90, 0]) if value['wall'] == 'wall_left_01' else ([w/2, y, x], [0, 90, 0]))
    return value, {'position': position, 'rotation_euler': rotation, 'scale': [1, 1, 1]}


def fit_estimated_windows(scene):
    """Project approximate openings onto disjoint wall intervals; preserve size and order."""
    for wall in ('wall_back_01', 'wall_left_01', 'wall_right_01'):
        windows = [o for o in scene['objects'] if o.get('placement_state') == 'fixed_estimated' and
                   o.get('fixed_placement_proposal', {}).get('wall') == wall]
        windows.sort(key=lambda o: o['fixed_placement_proposal']['horizontal_m'])
        span = scene['room']['size'][0 if wall == 'wall_back_01' else 2]
        widths = [o['fixed_placement_proposal']['width_m'] for o in windows]
        if sum(widths) + max(0, len(widths)-1)*.05 > span-.2:
            raise ValueError('Estimated windows too wide for wall; review window dimensions: ' + wall)
        centers, edge = [], -span/2+.1
        for o, width in zip(windows, widths):
            center = max(o['fixed_placement_proposal']['horizontal_m'], edge+width/2)
            centers.append(center)
            edge = center+width/2+.05
        if centers:
            shift = max(0, centers[-1]+widths[-1]/2-(span/2-.1))
            centers = [c-shift for c in centers]
        # A backward sweep enforces the left boundary even for widely separated initial proposals.
        for i in range(len(centers)-2, -1, -1):
            centers[i] = min(centers[i], centers[i+1]-(widths[i]+widths[i+1])/2-.05)
        if centers and centers[0]-widths[0]/2 < -span/2+.1-1e-6:
            raise ValueError('Window proposals cannot fit without re-estimating positions')
        for o, center in zip(windows, centers):
            proposal = o['fixed_placement_proposal']
            old = proposal['horizontal_m']
            if abs(old-center) > 1e-6:
                proposal.setdefault('model_horizontal_m', old)
                proposal['adjustment'] = 'Geometric separation within selected wall; design estimate, not measurement'
            proposal['horizontal_m'] = center
            o['transform']['position'][0 if wall == 'wall_back_01' else 2] = center


def repair_scene_supports(settings, scene, reference, image, audit, estimate_fixed=False):
    """Recheck legacy v1 support assignments without inventing or dropping inventory IDs."""
    scene = copy.deepcopy(scene)
    rows = [{'id': o['id'], 'category': o['category'], 'bbox': o['visual_evidence']['bbox_pixels']}
            for o in scene['objects'] if o.get('visual_evidence', {}).get('bbox_pixels')]
    by_id = {r['id']: r for r in rows}
    unresolved, handled = [], set()
    proposals = [o['fixed_placement_proposal'] for o in scene['objects'] if o.get('fixed_placement_proposal')]
    for obj in scene['objects']:
        oid = obj['id']
        if oid not in by_id:
            continue
        if obj['category'] == 'window_frame':
            handled.add(oid)
            obj['properties'].update(movable=False, interaction_required=False, close_observation_required=False)
            obj['interaction_required'] = False
            obj['support_id'] = 'not_applicable'
            obj['asset']['unity_type'] = 'procedural_mesh'
            if obj.get('placement_state') not in {'fixed_verified', 'fixed_estimated'}:
                if not estimate_fixed:
                    unresolved.append({'id': oid, 'reason': 'Fixed window needs --estimate-fixed or a verified anchor in stage 23'})
                    continue
                try:
                    proposal, transform = estimate_window(settings, obj, scene, reference, image, proposals, audit/(oid+'.window.json'))
                    proposals.append(proposal)
                    obj.update(transform=transform, placement_state='fixed_estimated', fixed_placement_proposal=proposal,
                               dimensions_m=[proposal['width_m'], proposal['height_m'], .1],
                               estimated_dimensions_m={'width': proposal['width_m'], 'height': proposal['height_m'], 'depth': .1})
                except ValueError as exc:
                    unresolved.append({'id': oid, 'reason': str(exc)})
            continue
        if obj['properties']['movable'] and obj.get('support_validation_version') != 3:
            handled.add(oid)
            print('Verify support: ' + oid, flush=True)
            try:
                support = verify_support(settings, by_id[oid], rows, reference, image, audit/(oid+'.support.json'))
                obj.update(support_id=support['support_id'], support_evidence=support['evidence'])
                if support['support_id'] == 'unknown':
                    unresolved.append({'id': oid, 'reason': support['evidence']})
                else:
                    obj['support_validation_version'] = 3
            except ValueError as exc:
                unresolved.append({'id': oid, 'reason': str(exc)})
    # Validate the support graph before calling it verified.
    supports = [{'source': o['id'], 'target': o['support_id'], 'relation': 'on', 'weight': 1}
                for o in scene['objects'] if o['properties']['movable'] and o.get('support_id') not in {None, 'unknown'}]
    try:
        fit_estimated_windows(scene)
        bridge.validate(dict(scene, constraints=supports))
    except ValueError as exc:
        unresolved.append({'id': 'support_graph', 'reason': str(exc)})
    other_pending = [u for u in scene['generation'].get('unresolved', []) if u.get('id') not in handled and u.get('id') != 'support_graph']
    scene['generation'].update(unresolved=other_pending + unresolved,
                               review_status='needs_review' if other_pending or unresolved else 'complete',
                               support_review_version=3)
    return bridge.normalise(scene)


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
        'support_id': {'enum': ['unknown', 'not_applicable', 'floor_01'] + ids},
        'mounting': {'enum': ['supported', 'wall_mounted', 'window', 'background', 'unknown']},
        'estimated_dimensions_m': DIMENSIONS, 'dimensions_plausible': {'type': 'boolean'},
        'description': TEXT, 'asset_prompt': TEXT})
    schema['properties']['support_evidence'] = TEXT  # Optional when copying a reviewed decision.
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
            if row['category'] == 'window_frame' and value['status'] == 'confirmed' and (
                    value['mounting'] != 'window' or value['representation'] != 'procedural'):
                raise ValueError('Detected window_frame MUST be procedural with mounting=window, never supported furniture')
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
            if value['status'] == 'confirmed' and value['mounting'] == 'supported':
                support = verify_support(settings, row, rows, reference, image, audit / (row['id'] + '.support.json'))
                value = dict(value, support_id=support['support_id'], support_evidence=support['evidence'])
        except ValueError as exc:
            unresolved.append({'id': row['id'], 'reason': str(exc)})
            continue
        decisions[row['id']] = value
        if value['status'] == 'part_or_duplicate' or value['status'] == 'outdoor_background':
            continue
        fixed = value['mounting'] in {'wall_mounted', 'window'}
        if (value['status'] != 'confirmed' or not value['dimensions_plausible'] or
            value['mounting'] == 'unknown' or
            (not fixed and value['representation'] == 'mesh' and value['support_id'] in {'unknown', 'not_applicable'})):
            unresolved.append({'id': row['id'], 'reason': value['reason'], 'review': value})
            continue
        if fixed and row['id'] not in anchors:
            unresolved.append({'id': row['id'], 'reason': 'Fixed transform required: --anchors in 23 or --estimate-fixed in 34 for windows', 'review': value})
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
                       support_validation_version=3, support_evidence=value.get('support_evidence'),
                       estimated_dimensions_m=value['estimated_dimensions_m'])
            obj['visual_evidence']['confidence_kind'] = 'unscored_placeholder'
            if value['mounting'] in {'wall_mounted', 'window'}:
                obj['transform'] = anchors.get(obj['id'], obj['transform'])
                obj['properties']['movable'] = False
                obj['properties']['interaction_required'] = False
                obj['interaction_required'] = False
                obj['placement_state'] = 'fixed_verified' if obj['id'] in anchors else 'fixed_pending'
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
        schema['properties']['constraints']['maxItems'] = 4
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
Emit only 1 to 4 essential relations, never enumerate the schema's choices. A verified on relation alone
is sufficient if other relationships are ambiguous. Never touch opposite walls simultaneously.
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
            check_relation_geometry(constraints + value['constraints'], scene)
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
