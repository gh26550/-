import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'scripts'))
import local_scene as m
import scene_review as r
from PIL import Image


class ReviewTests(unittest.TestCase):
    def scene(self, count=19):
        rows = [{'id': f'plant_{i:02}', 'name': 'plant', 'category': 'plant', 'representation': 'mesh',
                 'movable': True, 'description': 'potted plant', 'asset_prompt': 'potted plant',
                 'bbox': [i, 0, i+1, 2], 'dimensions_m': [.3, .5, .3], 'confidence': .5,
                 'evidence': 'visible pot'} for i in range(count)]
        scene = m.build_scene({'objects': rows, 'summary': 'plants', 'uncertainties': []}, 'test',
                              [6, 4, 6], {}, {'model': 'test'})
        for o in scene['objects'][5:]:
            o['support_id'] = 'floor_01'
        return scene

    def answer(self, oid, status='resolved'):
        return {'status': status, 'reason': 'pot touches floor', 'uncertainties': [], 'constraints': [] if status != 'resolved' else
                [{'source': oid, 'relation': 'on', 'target': 'floor_01', 'weight': 1, 'confidence': .8,
                  'evidence': 'pot touches floor', 'basis': 'observed'}]}

    def test_all_nineteen_sources_have_separate_requests_and_coverage(self):
        scene = self.scene()
        calls = []
        def answer(settings, prompt, schema, images, validate, audit):
            oid = schema['properties']['constraints']['items']['properties']['source']['enum'][0]
            calls.append(oid)
            value = self.answer(oid)
            m.jsonschema.validate(value, schema)
            validate(value)
            return value
        with tempfile.TemporaryDirectory() as temp, patch.object(r, 'cached_generate', side_effect=answer), patch.object(r, 'crop_image', return_value='crop'):
            result = r.generate_constraints({'model': 'test'}, scene, {'path': 'test'}, 'whole', Path(temp)/'out.json')
        self.assertEqual(result['status'], 'complete')
        self.assertEqual(len(result['constraints']), 19)
        self.assertTrue(all(calls.count(f'plant_{i:02}') == 2 for i in range(19)))

    def test_missing_object_persists_incomplete_and_blocks_export(self):
        scene = self.scene(2)
        scene['generation'].update(review_pipeline='per_object_v1', review_status='complete', unresolved=[])
        def answer(settings, prompt, schema, images, validate, audit):
            oid = schema['properties']['constraints']['items']['properties']['source']['enum'][0]
            value = self.answer(oid, 'needs_review' if oid == 'plant_01' else 'resolved')
            validate(value)
            return value
        with tempfile.TemporaryDirectory() as temp, patch.object(r, 'cached_generate', side_effect=answer), patch.object(r, 'crop_image', return_value='crop'):
            out = Path(temp)/'out.json'
            result = r.generate_constraints({'model': 'test'}, scene, {'path': 'test'}, 'whole', out)
            self.assertEqual(result['status'], 'incomplete')
            self.assertEqual(len(result['constraints']), 1)
            merged = m.bridge.normalise(scene, result)
            with self.assertRaises(ValueError):
                m.bridge.export_bundle(merged, Path(temp)/'bundle', temp)

    def test_scene_edit_invalidates_constraints(self):
        scene = self.scene(1)
        constraint_file = {'constraints': [], 'scene_fingerprint': r.fingerprint(scene)}
        scene['objects'][-1]['support_id'] = 'changed'
        with self.assertRaisesRegex(ValueError, 'Scene changed'):
            m.bridge.normalise(scene, constraint_file)

    def test_crop_and_overlap_do_not_merge_detections(self):
        rows = [{'id': 'a', 'bbox': [0, 0, 20, 20]}, {'id': 'b', 'bbox': [1, 1, 21, 21]}]
        self.assertEqual(r.overlap_candidates(rows[0], rows), ['b'])
        self.assertEqual(len(rows), 2)
        with tempfile.TemporaryDirectory() as temp:
            path = Path(temp)/'image.png'
            Image.new('RGB', (30, 30)).save(path)
            self.assertTrue(r.crop_image(path, rows[0]['bbox']))

    def test_cache_reuses_success_but_retries_uncertain_response(self):
        schema = m.obj_schema({'status': {'enum': ['resolved', 'needs_review']}})
        with tempfile.TemporaryDirectory() as temp, patch.object(r, 'generate', side_effect=[{'status': 'needs_review'}, {'status': 'resolved'}]) as call:
            for _ in range(3):
                r.cached_generate({}, 'p', schema, 'im', lambda x: None, Path(temp)/'audit.json')
            self.assertEqual(call.call_count, 2)

    def test_recognition_missing_support_and_fixed_anchor(self):
        detection = {'summary': 'shelf plant', 'uncertainties': [], 'objects': [
            {'id': 'shelf_01', 'name': 'shelf', 'category': 'shelf', 'bbox': [0, 0, 20, 10]},
            {'id': 'plant_01', 'name': 'plant', 'category': 'plant', 'bbox': [5, 1, 10, 8]}]}
        def decision(mounting, support):
            return {'status': 'confirmed', 'related_id': 'unknown', 'reason': 'visible support',
                    'representation': 'mesh', 'support_id': support, 'mounting': mounting,
                    'estimated_dimensions_m': {'width': 1, 'height': .1, 'depth': .3},
                    'dimensions_plausible': True, 'description': 'wood', 'asset_prompt': 'wood'}
        responses = [detection, detection, decision('wall_mounted', 'not_applicable'), decision('supported', 'shelf_01')]
        with tempfile.TemporaryDirectory() as temp, patch.object(r, 'crop_image', return_value='crop'), \
                patch.object(r, 'verify_support', return_value={'support_id': 'shelf_01', 'evidence': 'pot on shelf'}):
            with patch.object(r, 'cached_generate', side_effect=copy.deepcopy(responses)):
                scene = r.recognize({'model': 'test'}, {'path': 'x', 'width': 30, 'height': 30}, 'im', 'test', [6,4,6], Path(temp)/'scene.json')
            self.assertEqual(scene['generation']['review_status'], 'needs_review')
            anchor = {'shelf_01': {'position': [-2, 2, 0], 'rotation_euler': [0, 0, 0], 'scale': [1, 1, 1]}}
            with patch.object(r, 'cached_generate', side_effect=responses):
                scene = r.recognize({'model': 'test'}, {'path': 'x', 'width': 30, 'height': 30}, 'im', 'test', [6,4,6], Path(temp)/'scene.json', anchor)
            self.assertEqual(scene['generation']['review_status'], 'complete')
            self.assertFalse(scene['objects'][5]['properties']['movable'])
            self.assertEqual(scene['objects'][6]['support_id'], 'shelf_01')
            self.assertEqual(scene['objects'][5]['transform'], anchor['shelf_01'])

    def test_opposite_walls_rejected(self):
        constraints = [{'source': 'plant_00', 'relation': 'against', 'target': wall}
                       for wall in ('wall_left_01', 'wall_right_01')]
        with self.assertRaisesRegex(ValueError, 'opposite'):
            r.check_relation_geometry(constraints, self.scene(1))

    def test_distant_tv_stand_not_support_candidate(self):
        sofa = {'id': 'couch', 'category': 'sofa', 'bbox': [652,417,959,668]}
        stand = {'id': 'tv_stand', 'category': 'table', 'bbox': [48,463,319,640]}
        self.assertEqual(r.possible_supports(sofa, [stand, sofa]), [])

    def test_cached_success_revalidated_and_repaired(self):
        schema = m.obj_schema({'value': {'type': 'number'}})
        with tempfile.TemporaryDirectory() as temp, patch.object(r, 'generate', side_effect=[{'value': 0}, {'value': 1}]) as call:
            path = Path(temp)/'a.json'
            r.cached_generate({}, 'same', schema, 'im', lambda x: None, path)
            def strict(value):
                if value['value'] == 0:
                    raise ValueError('old acceptance bug')
            result = r.cached_generate({}, 'same', schema, 'im', strict, path)
            self.assertEqual(result['value'], 1)
            self.assertEqual(call.call_count, 2)

    def test_unknown_support_is_not_cached(self):
        schema = m.obj_schema({'support_id': m.TEXT, 'evidence': m.TEXT})
        with tempfile.TemporaryDirectory() as temp, patch.object(r, 'generate', return_value={'support_id': 'unknown', 'evidence': 'occluded'}) as call:
            for _ in range(2):
                r.cached_generate({}, 'p', schema, 'im', lambda x: None, Path(temp)/'audit.json')
            self.assertEqual(call.call_count, 2)

    def test_unrelated_pending_review_survives_support_repair(self):
        scene = self.scene(1)
        scene['objects'][-1]['properties']['movable'] = False
        scene['objects'][-1]['placement_state'] = 'fixed_pending'
        scene['generation']['unresolved'] = [{'id': 'plant_00', 'reason': 'fixed shelf needs transform'}]
        with tempfile.TemporaryDirectory() as temp:
            repaired = r.repair_scene_supports({}, scene, {}, 'im', Path(temp))
        self.assertEqual(repaired['generation']['unresolved'], scene['generation']['unresolved'])

    def test_repair_old_support_and_window_classification(self):
        scene = self.scene(2)
        scene['objects'][-1]['category'] = 'window_frame'
        scene['objects'][-1]['support_id'] = 'plant_00'
        scene['objects'][-2]['support_id'] = 'plant_01'
        with tempfile.TemporaryDirectory() as temp, patch.object(r, 'verify_support', return_value={'support_id': 'floor_01', 'evidence': 'on floor'}):
            fixed = r.repair_scene_supports({'model': 'test'}, scene, {}, 'image', Path(temp))
            self.assertFalse(fixed['objects'][-1]['properties']['movable'])
            self.assertEqual(fixed['objects'][-2]['support_id'], 'floor_01')
            self.assertEqual(fixed['generation']['review_status'], 'needs_review')
            self.assertTrue(fixed['generation']['unresolved'])

    def test_window_estimation_separates_overlapping_openings(self):
        scene = self.scene(1)
        obj = scene['objects'][-1]
        proposal = {'wall': 'wall_back_01', 'horizontal_m': 0, 'center_height_m': 2,
                    'width_m': 1, 'height_m': 2, 'evidence': 'back window'}
        def answer(settings, prompt, schema, images, validate, audit):
            raw = {'wall': 'wall_back_01', 'left_fraction': .4, 'right_fraction': .6,
                   'bottom_fraction': .25, 'top_fraction': .75, 'evidence': 'back window'}
            validate(raw)
            return raw
        with tempfile.TemporaryDirectory() as temp, patch.object(r, 'cached_generate', side_effect=answer), patch.object(r, 'crop_image', return_value='crop'):
            value, transform = r.estimate_window({}, obj, scene, {'path': 'x'}, 'im', [proposal], Path(temp)/'a.json')
            obj.update(placement_state='fixed_estimated', fixed_placement_proposal=value, transform=transform)
            other = copy.deepcopy(obj); other['id'] = 'other_window'
            scene['objects'].append(other)
            r.fit_estimated_windows(scene)
            self.assertGreaterEqual(other['transform']['position'][0]-obj['transform']['position'][0], value['width_m']+.049)


if __name__ == '__main__':
    unittest.main()
