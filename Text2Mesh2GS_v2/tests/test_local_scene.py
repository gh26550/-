import copy
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))
import local_scene as m


class LocalSceneTests(unittest.TestCase):
    def inventory(self):
        return {"summary": "Two visible chairs", "uncertainties": [], "objects": [
            {"id": oid, "name": "chair", "category": "chair", "representation": "mesh",
             "movable": True, "description": "wood chair", "asset_prompt": "wooden chair",
             "bbox": box, "dimensions_m": [.5, .9, .5], "confidence": .9, "evidence": "visible legs"}
            for oid, box in [("chair_01", [.1, .2, .4, .8]), ("chair_02", [.6, .2, .9, .8])]]}

    def scene(self):
        return m.build_scene(self.inventory(), "test", [6, 4, 6], {"path": "x", "sha256": "abc"}, {"model": "local"})

    def test_instances_and_provenance_survive_prepare(self):
        scene = self.scene()
        self.assertEqual([o["id"] for o in scene["objects"] if o["properties"]["movable"]], ["chair_01", "chair_02"])
        self.assertEqual(scene["reference_image"]["sha256"], "abc")
        self.assertEqual(m.bridge.normalise(scene), scene)

    def test_rug_is_not_silently_reclassified_as_procedural(self):
        data = self.inventory(); data["objects"][0]["category"] = "rug"
        scene = m.build_scene(data, "test", [6, 4, 6], {}, {"model": "local"})
        self.assertEqual(scene["objects"][5]["asset"]["unity_type"], "mesh_gameobject")

    def test_pixels_are_normalized_using_actual_image_size(self):
        data = self.inventory(); data["objects"] = data["objects"][:1]
        data["objects"][0]["bbox"] = [100, 50, 300, 150]
        scene = m.build_scene(data, "test", [6, 4, 6], {"width": 400, "height": 200}, {"model": "local"})
        self.assertEqual(scene["objects"][5]["visual_evidence"]["bbox"], [.25, .25, .75, .75])
        data["objects"][0]["bbox"][2] = 401
        with self.assertRaises(ValueError): m.check_inventory(data, {"width": 400, "height": 200})

    def test_identical_visual_instances_rejected(self):
        data = self.inventory(); data["objects"][1]["bbox"] = data["objects"][0]["bbox"]
        with self.assertRaises(ValueError): m.check_inventory(data)

    def test_constraints_from_another_image_rejected(self):
        with self.assertRaises(ValueError):
            m.bridge.normalise(self.scene(), {"constraints": [], "reference_image": {"sha256": "different"}})

    def test_duplicate_instance_and_invalid_bbox_rejected(self):
        data = self.inventory(); data["objects"][1]["id"] = "chair_01"
        with self.assertRaises(ValueError): m.check_inventory(data)
        data = self.inventory(); data["objects"][0]["bbox"] = [.9, .1, .2, .3]
        with self.assertRaises(ValueError): m.check_inventory(data)

    def test_missing_image_is_error(self):
        with self.assertRaises(FileNotFoundError): m.image_info("nonexistent-image-12345.png")

    def test_edge_overshoot_and_duplicate_are_audited(self):
        data = self.inventory(); data["objects"] = data["objects"][:1]
        data["objects"][0]["bbox"] = [864, 397, 1035, 629]
        other = copy.deepcopy(data["objects"][0]); other["id"] = "chair_02"
        data["objects"].append(other)
        cleaned, changes = m.normalize_inventory(data, {"width": 1024, "height": 768})
        self.assertEqual(len(cleaned["objects"]), 1)
        self.assertEqual(cleaned["objects"][0]["bbox"], [864, 397, 1024, 629])
        self.assertEqual(data["objects"][0]["bbox"][2], 1035)
        self.assertIn("exact_duplicate", [c["kind"] for c in changes])
        m.check_inventory(cleaned, {"width": 1024, "height": 768})

    def test_large_overshoot_and_duplicate_ids_still_fail(self):
        data = self.inventory(); data["objects"][0]["bbox"] = [800, 20, 1200, 700]
        cleaned, _ = m.normalize_inventory(data, {"width": 1024, "height": 768})
        with self.assertRaises(ValueError): m.check_inventory(cleaned, {"width": 1024, "height": 768})
        data = self.inventory(); data["objects"].append(copy.deepcopy(data["objects"][0]))
        cleaned, _ = m.normalize_inventory(data, {"width": 1024, "height": 768})
        with self.assertRaises(ValueError): m.check_inventory(cleaned, {"width": 1024, "height": 768})

    def test_semantic_repair_not_rules_fallback(self):
        bad = self.inventory(); bad["objects"][0]["bbox"] = [1, 1, 0, 0]
        good = self.inventory()
        with tempfile.TemporaryDirectory() as temp, patch.object(m, "ask", side_effect=[bad, good]) as call:
            result = m.generate({"model": "local", "repair_attempts": 1}, "prompt", m.INVENTORY, "image",
                                m.check_inventory, Path(temp) / "audit.json")
            self.assertEqual(result, good)
            self.assertEqual(call.call_count, 2)
            self.assertIn("Bounding box", str(call.call_args.args[-1]))

    def test_no_valid_output_after_failed_repair(self):
        with tempfile.TemporaryDirectory() as temp, patch.object(m, "ask", return_value={}):
            with self.assertRaises(ValueError):
                m.generate({"model": "local", "repair_attempts": 1}, "prompt", m.INVENTORY, "image",
                           m.check_inventory, Path(temp) / "audit.json")

    def test_uncovered_object_rejected(self):
        with self.assertRaises(ValueError): m.check_constraints({"constraints": []}, self.scene())

    def test_constraint_schema_rejects_fixed_sources_before_semantic_check(self):
        data = {"uncertainties": [], "constraints": [{"source": "wall_left_01", "relation": "near", "target": "chair_01",
                "weight": 1, "confidence": .8, "evidence": "test"}]}
        with self.assertRaises(m.jsonschema.ValidationError):
            m.jsonschema.validate(data, m.constraint_schema(self.scene()))

    def test_export_uses_model_prompt_and_excludes_room_anchors(self):
        with tempfile.TemporaryDirectory() as temp:
            m.bridge.export_bundle(self.scene(), temp, temp)
            with (Path(temp) / "mesh_object_prompts.csv").open(encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            self.assertEqual(len(rows), 2)
            self.assertEqual(rows[0]["object_prompt"], "wooden chair")

    def test_request_contains_image_and_schema_and_unloads_model(self):
        class Response:
            def __enter__(self): return self
            def __exit__(self, *args): pass
            def read(self): return json.dumps({"done": True, "message": {"content": "{}"}}).encode()
        with patch.object(m.urllib.request, "build_opener") as opener:
            opener.return_value.open.return_value = Response()
            m.ask({"model": "local", "url": "http://127.0.0.1:11434"}, "test", m.INVENTORY, "encoded-image")
            body = json.loads(opener.return_value.open.call_args.args[0].data)
            self.assertEqual(body["messages"][1]["images"], ["encoded-image"])
            self.assertEqual(body["format"], m.INVENTORY)
            self.assertEqual(body["keep_alive"], 0)


if __name__ == "__main__":
    unittest.main()
