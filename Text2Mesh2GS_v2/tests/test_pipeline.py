import argparse
import copy
import importlib.util
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
import pipeline as p
import render_settings as rs


class PipelineTests(unittest.TestCase):
    def scene(self):
        return p.normalise(p.read(ROOT / "examples/layout_legacy.json"), p.read(ROOT / "examples/constraints_legacy.json"))

    def test_legacy_normalisation_and_duplicate_constraint(self):
        scene = self.scene()
        self.assertEqual(len(scene["objects"]), 14)
        self.assertEqual(len(scene["constraints"]), 18)
        self.assertTrue(all("weight" in c for c in scene["constraints"]))
        self.assertEqual(p.normalise(scene), scene)

    def test_mismatched_scene(self):
        with self.assertRaises(ValueError):
            p.normalise(p.read(ROOT / "examples/layout_legacy.json"), {"scene_id": "other", "constraints": []})

    def test_missing_object_is_error(self):
        scene = self.scene()
        scene["constraints"][0]["source"] = "unknown"
        with self.assertRaises(ValueError): p.validate(scene)

    def test_negative_weight_is_error(self):
        scene = self.scene(); scene["constraints"][0]["weight"] = -1
        with self.assertRaises(ValueError): p.validate(scene)

    def test_nonfinite_scale_is_error(self):
        scene = self.scene(); scene["objects"][0]["transform"]["scale"][0] = float("nan")
        with self.assertRaises(ValueError): p.validate(scene)

    def test_support_cycle(self):
        scene = self.scene()
        scene["constraints"].append({"source": "table_01", "target": "vase_01", "relation": "on", "weight": 1})
        with self.assertRaisesRegex(ValueError, "Cyclic"): p.validate(scene)

    def test_support_before_child(self):
        order = p.support_order(self.scene())
        self.assertLess(order.index("table_01"), order.index("vase_01"))

    def test_contradictory_relations(self):
        scene = self.scene()
        scene["constraints"].append({"source": "sofa_01", "target": "table_01", "relation": "left of", "weight": 1})
        with self.assertRaisesRegex(ValueError, "Contradictory"): p.validate(scene)

    def test_no_hybrid(self):
        scene = self.scene(); scene["objects"][0]["representation"] = "hybrid"
        with self.assertRaises(ValueError): p.validate(scene)

    def test_bundle_paths_agree(self):
        scene = self.scene()
        with tempfile.TemporaryDirectory() as temp:
            p.export_bundle(scene, temp, temp)
            layout = p.read(Path(temp) / "layout_for_unity.json")
            jobs = p.read(Path(temp) / "gaussian_import_jobs.json")
            target = next(o for o in layout["unity_objects"] if o["id"] == jobs["jobs"][0]["object_id"])
            self.assertEqual(target["asset"]["asset_path"], jobs["jobs"][0]["generation"]["output_ply"])
            self.assertFalse(p.read(Path(temp) / "validation.json")["assets_generated"])

    def test_plan_flags_and_portable_validation(self):
        jobs = {"jobs": [{"object_id": "outside", "status": "needs_gaussian_conversion", "source_path": "source.glb",
                          "generation": {"multiview_output_dir": "mv", "gaussian_output_dir": "gs", "output_ply": "out.ply"}}]}
        config = p.read(ROOT / "configs/gaussian.json")
        config["use_zbuffer_visibility"] = True
        plan = p.gaussian_plan(jobs, config, ".")
        commands = plan[0]["commands"]
        self.assertIn("--use_anisotropic_init", commands[1])
        self.assertIn("--use_zbuffer_visibility", commands[1])
        self.assertEqual(sum("--no_zbuffer_visibility" in c for c in commands), 3)
        self.assertIn("--scale_max", commands[-1])

    def test_image_only_jobs_fail_instead_of_running_blender(self):
        with self.assertRaisesRegex(ValueError, "provide an existing GLB"):
            p.gaussian_plan({"jobs": [{"object_id": "x", "status": "needs_gaussian_training"}]}, p.read(ROOT / "configs/gaussian.json"), ".")

    def test_checkpoint_settings_and_explicit_override(self):
        args = argparse.Namespace(scale_min=.1, scale_max=.5, image_downscale=2, use_zbuffer_visibility=False, no_zbuffer_visibility=True)
        rs.inherit_settings(args, {"render_settings": {"scale_min": .001, "scale_max": .01, "use_zbuffer_visibility": True}}, ["--scale_max=.5"])
        self.assertEqual(args.scale_min, .001)
        self.assertEqual(args.scale_max, .5)
        self.assertFalse(args.use_zbuffer_visibility)

    def test_invalid_scale_settings(self):
        with self.assertRaises(ValueError): rs.validate_settings({"scale_min": 2, "scale_max": 1})


if __name__ == "__main__":
    unittest.main()
