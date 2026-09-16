using System;
using UnityEngine;

namespace Text2Mesh2GS.V2
{
    [Serializable] public class LayoutData
    {
        public string schema_version;
        public string scene_id;
        public RoomData room;
        public ObjectData[] unity_objects;
    }
    [Serializable] public class RoomData { public float[] size; }
    [Serializable] public class ObjectData
    {
        public string id, name, category, representation;
        public bool movable, background_only;
        public PoseData transform;
        public AssetData asset;
        public float[] forward_axis;
    }
    [Serializable] public class PoseData { public float[] position, rotation_euler, scale; }
    [Serializable] public class AssetData { public string unity_type, asset_path; }
    [Serializable] public class ConstraintFile
    {
        public string schema_version, scene_id;
        public ConstraintData[] constraints;
    }
    [Serializable] public class ConstraintData
    {
        public string type, relation, source, target;
        public float weight = 1;
    }
    [Serializable] public class ConstraintResult
    {
        public string source, target, relation, status;
        public float weight, measured, tolerance, satisfaction, contribution;
    }
    [Serializable] public class ObjectResult
    {
        public string id, status;
        public float[] position, rotation_euler, scale;
    }
    [Serializable] public class PlacementReport
    {
        public string schema_version = "2.0", scene_id, created_at;
        public string status, score_definition = "100*sum(weight*satisfaction)/sum(expected furniture weight); failures retained in denominator";
        public bool physically_feasible, semantic_complete;
        public int seed, trial, expected_constraints, evaluated_constraints, passed_constraints, missing_constraints, unsupported_constraints;
        public int physical_violations, missing_objects;
        public float score_100, coverage_100, weighted_sum, physical_error;
        public ConstraintResult[] constraints;
        public ObjectResult[] objects;
    }
}
