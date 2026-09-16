using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;

namespace Text2Mesh2GS.V2
{
    /// <summary>Room-local, constraint-only finite search with support groups and final feasibility audit.</summary>
    public class SemanticScenePlacerV2 : MonoBehaviour
    {
        public TextAsset layoutJson, constraintsJson;
        [Tooltip("Room origin at floor centre; local X=right, Y=up, front=-Z. Unit scale required.")]
        public Transform roomFrame;
        public Transform movableRoot, referenceRoot;
        public bool applyJsonScale = false;
        public bool deterministicStart = true;
        public int seed = 12345, restartCount = 1, refinementSweeps = 3, candidateLimit = 400;
        public float candidateStep = .35f, nearDistance = 1.25f, centreTolerance = .75f;
        public Vector2 roomCentre = Vector2.zero;
        public float contactEpsilon = .005f, boundaryTolerance = .03f, overlapVolumeTolerance = .0005f;
        public float wallGap = .08f, angleTolerance = 10f, angleHalfScoreExcess = 35f;
        public PlacementReport lastReport;

        LayoutData layout;
        ConstraintFile input;
        readonly Dictionary<string, Transform> objects = new Dictionary<string, Transform>();
        readonly Dictionary<string, string> supports = new Dictionary<string, string>();
        readonly List<ObjectData> movable = new List<ObjectData>();
        readonly List<Bounds> obstacles = new List<Bounds>();
        struct Pose { public Vector3 position, scale; public Quaternion rotation; }
        struct Candidate { public Vector3 centre; public Quaternion rotation; }

        bool Load()
        {
            objects.Clear(); supports.Clear(); movable.Clear(); obstacles.Clear();
            if (layoutJson == null || constraintsJson == null || roomFrame == null || movableRoot == null || referenceRoot == null)
                throw new InvalidOperationException("Assign layout, constraints, roomFrame, movableRoot and referenceRoot.");
            if ((roomFrame.lossyScale - Vector3.one).sqrMagnitude > 1e-6f)
                throw new InvalidOperationException("roomFrame must have unit world scale (metres).");
            layout = JsonUtility.FromJson<LayoutData>(layoutJson.text);
            input = JsonUtility.FromJson<ConstraintFile>(constraintsJson.text);
            if (layout.schema_version != "2.0" || input.schema_version != "2.0" || input.scene_id != layout.scene_id)
                throw new InvalidOperationException("Run pipeline.py prepare: schema 2.0 and matching scene IDs required.");
            if (layout.unity_objects == null || input.constraints == null || layout.room == null || layout.room.size == null || layout.room.size.Length != 3)
                throw new InvalidOperationException("Invalid input structure.");
            if (layout.room.size.Any(v => v <= 0 || float.IsNaN(v) || float.IsInfinity(v)))
                throw new InvalidOperationException("Invalid room dimensions.");
            if (layout.unity_objects.Select(o => o.id).Distinct().Count() != layout.unity_objects.Length)
                throw new InvalidOperationException("Duplicate layout IDs.");
            var all = movableRoot.GetComponentsInChildren<Transform>(true)
                .Concat(referenceRoot.GetComponentsInChildren<Transform>(true)).Distinct().ToArray();
            foreach (var item in layout.unity_objects)
            {
                var matches = all.Where(t => t.GetComponent<SceneObjectIdV2>() != null && t.GetComponent<SceneObjectIdV2>().objectId == item.id).ToArray();
                if (matches.Length == 0) matches = all.Where(t => t.name == item.id).ToArray();
                if (matches.Length > 1) throw new InvalidOperationException("Ambiguous ID: " + item.id);
                if (matches.Length == 1) objects[item.id] = matches[0];
                if (item.movable && !item.background_only && item.representation == "mesh") movable.Add(item);
            }
            foreach (var c in input.constraints)
            {
                if (c == null || string.IsNullOrEmpty(c.source) || string.IsNullOrEmpty(c.relation)) throw new InvalidOperationException("Invalid constraint.");
                c.relation = c.relation.Trim().ToLowerInvariant();
                if (c.weight < 0 || float.IsNaN(c.weight) || float.IsInfinity(c.weight)) throw new InvalidOperationException("Invalid weight.");
                if (c.relation == "on")
                {
                    if (supports.ContainsKey(c.source) && supports[c.source] != c.target) throw new InvalidOperationException("Multiple supports: " + c.source);
                    supports[c.source] = c.target;
                }
            }
            foreach(var a in movable.Where(o=>objects.ContainsKey(o.id)))
                foreach(var b in movable.Where(o=>o.id!=a.id&&objects.ContainsKey(o.id)))
                    if(objects[a.id].IsChildOf(objects[b.id]))throw new InvalidOperationException("Movable roots must not be nested: "+a.id);
            foreach (var item in movable) SupportDepth(item.id, new HashSet<string>());
            // No room-level enclosing box: use each visible wall/frame piece independently.
            foreach (var renderer in referenceRoot.GetComponentsInChildren<Renderer>(true))
            {
                var marker = renderer.GetComponentInParent<SceneObjectIdV2>();
                var item = marker == null ? null : layout.unity_objects.FirstOrDefault(o => o.id == marker.objectId);
                string n = renderer.name.ToLowerInvariant();
                if ((item != null && (item.category == "wall" || item.category == "window_frame")) || n.Contains("wall") || n.Contains("frame"))
                    if (PlacementBounds.TryGet(renderer.transform, roomFrame, out var bounds)) obstacles.Add(bounds);
            }
            candidateStep = Mathf.Max(.05f, candidateStep); candidateLimit = Mathf.Max(8, candidateLimit);
            return true;
        }

        int SupportDepth(string id, HashSet<string> visited)
        {
            if (!visited.Add(id)) throw new InvalidOperationException("Cyclic on constraints: " + id);
            if (!supports.TryGetValue(id, out var parent)) return 0;
            return 1 + SupportDepth(parent, visited);
        }
        bool IsDescendant(string id, string parent)
        {
            while (supports.TryGetValue(id, out var target)) { if (target == parent) return true; id = target; }
            return false;
        }
        bool BoundsFor(string id, out Bounds b)
        {
            b = new Bounds();
            return !string.IsNullOrEmpty(id) && objects.TryGetValue(id, out var t) && PlacementBounds.TryGet(t, roomFrame, out b);
        }
        static bool IsCentre(string rel) { return rel == "center of room" || rel == "near room center" || rel == "center of scene"; }
        static float Gap(Bounds a, Bounds b)
        {
            float x = Mathf.Max(0, Mathf.Max(a.min.x - b.max.x, b.min.x - a.max.x));
            float z = Mathf.Max(0, Mathf.Max(a.min.z - b.max.z, b.min.z - a.max.z));
            return Mathf.Sqrt(x*x + z*z);
        }
        static float Volume(Bounds a, Bounds b)
        {
            Vector3 d = Vector3.Min(a.max,b.max)-Vector3.Max(a.min,b.min);
            return Mathf.Max(0,d.x)*Mathf.Max(0,d.y)*Mathf.Max(0,d.z);
        }
        static float Overhang(Bounds a, Bounds b)
        {
            return Mathf.Max(0,b.min.x-a.min.x)+Mathf.Max(0,a.max.x-b.max.x)+Mathf.Max(0,b.min.z-a.min.z)+Mathf.Max(0,a.max.z-b.max.z);
        }
        Vector3 Forward(string id)
        {
            Transform t = objects[id];
            var marker = t.GetComponent<SceneObjectIdV2>();
            var item = layout.unity_objects.First(o => o.id == id);
            var axis = marker != null ? marker.forwardAxis : item.forward_axis != null && item.forward_axis.Length == 3
                ? new Vector3(item.forward_axis[0],item.forward_axis[1],item.forward_axis[2]) : Vector3.forward;
            if (axis.sqrMagnitude < 1e-8f) axis = Vector3.forward;
            return roomFrame.InverseTransformDirection(t.TransformDirection(axis.normalized));
        }
        float AgainstDistance(Bounds b, string target)
        {
            if (!BoundsFor(target, out var wall)) return float.PositiveInfinity;
            Vector3 normal = roomFrame.InverseTransformDirection(objects[target].forward).normalized;
            float a = ProjectExtent(b,normal), w = ProjectExtent(wall,normal);
            return Mathf.Abs(Mathf.Abs(Vector3.Dot(b.center-wall.center,normal))-a-w);
        }
        static float ProjectExtent(Bounds b, Vector3 n) { return Mathf.Abs(n.x)*b.extents.x+Mathf.Abs(n.y)*b.extents.y+Mathf.Abs(n.z)*b.extents.z; }

        ConstraintResult Evaluate(ConstraintData c)
        {
            var r = new ConstraintResult { source=c.source, target=c.target, relation=c.relation, weight=c.weight, status="missing_source" };
            if (!BoundsFor(c.source,out var a)) return r;
            bool centre = IsCentre(c.relation);
            Bounds b = new Bounds();
            if (!centre && !BoundsFor(c.target,out b)) { r.status="missing_target"; return r; }
            float half = 1;
            switch(c.relation)
            {
                case "on": r.measured=Mathf.Max(Mathf.Abs(a.min.y-b.max.y)/.03f,Overhang(a,b)/.05f); r.tolerance=1; half=1; break;
                case "near": r.measured=Gap(a,b); r.tolerance=nearDistance; half=nearDistance; break;
                case "face to":
                    Vector3 dir=b.center-a.center; dir.y=0;
                    Vector3 forward=Forward(c.source); forward.y=0;
                    if(dir.sqrMagnitude<1e-8f || forward.sqrMagnitude<1e-8f) { r.status="undefined_direction"; return r; }
                    r.measured=Vector3.Angle(forward,dir); r.tolerance=angleTolerance; half=angleHalfScoreExcess; break;
                case "against": r.measured=AgainstDistance(a,c.target); r.tolerance=wallGap+.05f; half=.25f; break;
                case "left of": r.measured=Mathf.Max(0,a.center.x-b.center.x+.1f); break;
                case "right of": r.measured=Mathf.Max(0,b.center.x-a.center.x+.1f); break;
                case "in front of": r.measured=Mathf.Max(0,a.center.z-b.center.z+.1f); break;
                case "behind": r.measured=Mathf.Max(0,b.center.z-a.center.z+.1f); break;
                case "side of": r.measured=Mathf.Max(0,Mathf.Max(.4f,b.extents.x)-Mathf.Abs(a.center.x-b.center.x)); break;
                default:
                    if(!centre) { r.status="unsupported"; return r; }
                    r.measured=Vector2.Distance(new Vector2(a.center.x,a.center.z),roomCentre); r.tolerance=centreTolerance; half=centreTolerance; break;
            }
            r.status=r.measured<=r.tolerance+1e-6f ? "ok" : "unmet";
            r.satisfaction=(float)PlacementMath.Satisfaction(r.status=="ok"?0:r.measured,r.tolerance,half);
            r.contribution=100*r.weight*r.satisfaction;
            return r;
        }

        void Physical(PlacementReport report)
        {
            foreach(var item in movable)
            {
                if(!BoundsFor(item.id,out var b)) { report.missing_objects++; continue; }
                Vector3 room=new Vector3(layout.room.size[0],layout.room.size[1],layout.room.size[2]);
                float boundary=Mathf.Max(0,-room.x/2-b.min.x)+Mathf.Max(0,b.max.x-room.x/2)
                    +Mathf.Max(0,-room.z/2-b.min.z)+Mathf.Max(0,b.max.z-room.z/2)+Mathf.Max(0,-b.min.y)+Mathf.Max(0,b.max.y-room.y);
                if(boundary>boundaryTolerance) { report.physical_violations++; report.physical_error+=boundary; }
                float surface = 0;
                if(supports.TryGetValue(item.id,out var support) && BoundsFor(support,out var sb)) surface=sb.max.y;
                float floorError=Mathf.Abs(b.min.y-surface);
                if(floorError>.03f) { report.physical_violations++; report.physical_error+=floorError; }
                foreach(var obstacle in obstacles)
                {
                    float volume=Volume(b,obstacle);
                    if(volume>overlapVolumeTolerance) { report.physical_violations++; report.physical_error+=volume; }
                }
            }
            for(int i=0;i<movable.Count;i++) for(int j=i+1;j<movable.Count;j++)
            {
                string ai=movable[i].id, bi=movable[j].id;
                if(!BoundsFor(ai,out var a)||!BoundsFor(bi,out var b)) continue;
                // Contact exception only for a supported, contained pair; applied symmetrically.
                bool contact=(supports.TryGetValue(ai,out var ap)&&ap==bi&&Mathf.Abs(a.min.y-b.max.y)<=.03f&&Overhang(a,b)<=.05f)
                    ||(supports.TryGetValue(bi,out var bp)&&bp==ai&&Mathf.Abs(b.min.y-a.max.y)<=.03f&&Overhang(b,a)<=.05f);
                if(contact) continue;
                float volume=Volume(a,b);
                if(volume>overlapVolumeTolerance) { report.physical_violations++; report.physical_error+=volume; }
            }
        }
        PlacementReport Audit()
        {
            var report=new PlacementReport { scene_id=layout.scene_id, seed=seed, created_at=DateTime.UtcNow.ToString("o") };
            var ids=new HashSet<string>(movable.Select(o=>o.id));
            report.constraints=input.constraints.Where(c=>ids.Contains(c.source)).Select(Evaluate).ToArray();
            report.expected_constraints=report.constraints.Length;
            float weight=0;
            foreach(var r in report.constraints)
            {
                weight+=r.weight; report.weighted_sum+=r.contribution;
                if(r.status=="ok"||r.status=="unmet") report.evaluated_constraints++;
                if(r.status=="ok") report.passed_constraints++;
                if(r.status.StartsWith("missing")) report.missing_constraints++;
                if(r.status=="unsupported") report.unsupported_constraints++;
            }
            report.score_100=weight>0?report.weighted_sum/weight:0;
            report.coverage_100=report.expected_constraints>0?100f*report.evaluated_constraints/report.expected_constraints:0;
            Physical(report);
            report.physically_feasible=report.physical_violations==0&&report.missing_objects==0;
            report.semantic_complete=report.expected_constraints>0&&report.passed_constraints==report.expected_constraints;
            report.status=report.physically_feasible&&report.semantic_complete?"success":"incomplete";
            report.objects=movable.Select(o=> {
                bool found=objects.TryGetValue(o.id,out var t);
                bool bounds=BoundsFor(o.id,out var unused);
                return new ObjectResult { id=o.id,status=!found?"missing_object":!bounds?"missing_bounds":"present",
                    position=found?Values(roomFrame.InverseTransformPoint(t.position)):null,
                    rotation_euler=found?Values((Quaternion.Inverse(roomFrame.rotation)*t.rotation).eulerAngles):null,
                    scale=found?Values(t.localScale):null };
            }).ToArray();
            return report;
        }
        static float[] Values(Vector3 p) { return new[]{p.x,p.y,p.z}; }
        static bool Better(PlacementReport a, PlacementReport b)
        {
            return PlacementMath.Better(a.physical_violations+a.missing_objects,a.physical_error,a.weighted_sum,
                b.physical_violations+b.missing_objects,b.physical_error,b.weighted_sum);
        }
        Dictionary<string,Pose> Capture()
        {
            return movable.Where(o=>objects.ContainsKey(o.id)).ToDictionary(o=>o.id,o=>new Pose {
                position=objects[o.id].position,rotation=objects[o.id].rotation,scale=objects[o.id].localScale });
        }
        void Restore(Dictionary<string,Pose> poses)
        {
            foreach(var p in poses) { var t=objects[p.Key]; t.SetPositionAndRotation(p.Value.position,p.Value.rotation); t.localScale=p.Value.scale; }
        }
        Quaternion RotationFor(ObjectData item, Vector3 centre)
        {
            var face=input.constraints.FirstOrDefault(c=>c.source==item.id&&c.relation=="face to");
            Vector3 dir=Vector3.forward;
            if(face!=null&&BoundsFor(face.target,out var target)) { dir=target.center-centre; dir.y=0; }
            else
            {
                var against=input.constraints.FirstOrDefault(c=>c.source==item.id&&c.relation=="against");
                if(against!=null&&objects.TryGetValue(against.target,out var wall))
                { dir=roomFrame.InverseTransformDirection(wall.forward); if(BoundsFor(against.target,out var wb)&&Vector3.Dot(-wb.center,dir)<0)dir=-dir; dir.y=0; }
            }
            if(dir.sqrMagnitude<1e-8f)dir=Vector3.forward;
            var t=objects[item.id]; var marker=t.GetComponent<SceneObjectIdV2>();
            Vector3 local=marker!=null?marker.forwardAxis:item.forward_axis!=null&&item.forward_axis.Length==3?new Vector3(item.forward_axis[0],item.forward_axis[1],item.forward_axis[2]):Vector3.forward;
            if(local.sqrMagnitude<1e-8f)local=Vector3.forward;
            return roomFrame.rotation*Quaternion.LookRotation(dir.normalized,Vector3.up)*Quaternion.FromToRotation(local.normalized,Vector3.forward);
        }
        List<Candidate> Candidates(ObjectData item)
        {
            var list=new List<Candidate>();
            if(!BoundsFor(item.id,out var source))return list;
            Action<Vector3> add=p=>list.Add(new Candidate { centre=p,rotation=RotationFor(item,p) });
            if(supports.TryGetValue(item.id,out var support)&&BoundsFor(support,out var sb))
            {
                for(int x=-2;x<=2;x++)for(int z=-2;z<=2;z++)add(sb.center+new Vector3(x*sb.extents.x*.4f,0,z*sb.extents.z*.4f));
                return list;
            }
            add(new Vector3(roomCentre.x,source.center.y,roomCentre.y));
            // Both incoming and outgoing neighbours contribute candidates.
            foreach(var c in input.constraints.Where(c=>c.source==item.id||c.target==item.id))
            {
                string other=c.source==item.id?c.target:c.source;
                if(!BoundsFor(other,out var b))continue;
                for(int angle=0;angle<16;angle++)foreach(float gap in new[]{.1f,.5f,1f})
                {
                    float rad=angle*Mathf.PI/8;
                    add(b.center+new Vector3(Mathf.Cos(rad)*(b.extents.x+source.extents.x+gap),0,Mathf.Sin(rad)*(b.extents.z+source.extents.z+gap)));
                }
            }
            float width=layout.room.size[0], depth=layout.room.size[2];
            for(float x=-width/2+source.extents.x+wallGap;x<=width/2-source.extents.x;x+=candidateStep)
                for(float z=-depth/2+source.extents.z+wallGap;z<=depth/2-source.extents.z;z+=candidateStep)add(new Vector3(x,source.center.y,z));
            list=list.GroupBy(c=>Mathf.RoundToInt(c.centre.x*100)+":"+Mathf.RoundToInt(c.centre.z*100)).Select(g=>g.First()).ToList();
            if(list.Count<=candidateLimit)return list;
            // Evenly sample the whole pool; do not preferentially discard one relation family.
            return Enumerable.Range(0,candidateLimit).Select(i=>list[(int)((long)i*list.Count/candidateLimit)]).ToList();
        }
        void PlaceGroup(ObjectData item,Candidate candidate,Dictionary<string,Pose> original)
        {
            Transform t=objects[item.id];
            t.rotation=candidate.rotation;
            if(!BoundsFor(item.id,out var b))return;
            Vector3 centre=candidate.centre;
            float y=0;
            if(supports.TryGetValue(item.id,out var parent)&&BoundsFor(parent,out var support))y=support.max.y+contactEpsilon;
            centre.y=y+b.extents.y;
            t.position+=roomFrame.TransformVector(centre-b.center);
            Quaternion delta=t.rotation*Quaternion.Inverse(original[item.id].rotation);
            foreach(var child in movable.Where(o=>IsDescendant(o.id,item.id)&&objects.ContainsKey(o.id)))
            {
                Pose p=original[child.id];
                objects[child.id].SetPositionAndRotation(t.position+delta*(p.position-original[item.id].position),delta*p.rotation);
            }
        }

        [ContextMenu("Apply Semantic Layout V2")]
        public void ApplyLayout()
        {
            try
            {
                Load();
                var initial=Capture();
                try
                {
                    if(applyJsonScale)foreach(var item in movable.Where(o=>objects.ContainsKey(o.id)))
                    {
                        if(item.transform==null||item.transform.scale==null||item.transform.scale.Length!=3)throw new InvalidOperationException("Missing scale: "+item.id);
                        float[] s=item.transform.scale; objects[item.id].localScale=new Vector3(s[0],s[1],s[2]);
                    }
                    // Fixed start is independent of layout positions/rotations and Unity starting furniture poses.
                    if(deterministicStart)
                    {
                        foreach(var item in movable.Where(o=>objects.ContainsKey(o.id)))objects[item.id].SetPositionAndRotation(roomFrame.position,roomFrame.rotation);
                        foreach(var item in movable.OrderBy(o=>SupportDepth(o.id,new HashSet<string>())))
                            if(objects.ContainsKey(item.id))PlaceGroup(item,new Candidate { centre=Vector3.zero,rotation=RotationFor(item,Vector3.zero) },Capture());
                    }
                    var start=Capture(); Dictionary<string,Pose> best=null; PlacementReport bestReport=null;
                    var random=new System.Random(seed);
                    for(int trial=0;trial<Mathf.Max(1,restartCount);trial++)
                    {
                        Restore(start);
                        var order=movable.OrderBy(o=>SupportDepth(o.id,new HashSet<string>()))
                            .ThenByDescending(o=>input.constraints.Count(c=>c.target==o.id)).ThenBy(o=>trial==0?0:random.Next())
                            .ThenBy(o=>o.id,StringComparer.Ordinal).ToList();
                        for(int sweep=0;sweep<=Mathf.Max(0,refinementSweeps);sweep++)
                        {
                            bool changed=false;
                            foreach(var item in order)
                            {
                                if(!objects.ContainsKey(item.id))continue;
                                var saved=Capture(); var chosen=saved; var current=Audit();
                                foreach(var candidate in Candidates(item))
                                {
                                    Restore(saved); PlaceGroup(item,candidate,saved);
                                    var candidateReport=Audit();
                                    if(Better(candidateReport,current)) { current=candidateReport; chosen=Capture(); changed=true; }
                                }
                                Restore(chosen);
                            }
                            if(!changed)break;
                        }
                        var report=Audit(); report.trial=trial;
                        if(bestReport==null||Better(report,bestReport)) { bestReport=report;best=Capture(); }
                    }
                    Restore(best); lastReport=Audit(); lastReport.trial=bestReport.trial;
                    Debug.Log($"[Placement V2] {lastReport.status}: score={lastReport.score_100:F1}, coverage={lastReport.coverage_100:F1}%, physical violations={lastReport.physical_violations}, missing={lastReport.missing_objects}");
                }
                catch { Restore(initial); throw; }
            }
            catch(Exception e) { Debug.LogError("[Placement V2] "+e.Message); }
        }
        [ContextMenu("Audit Current Layout V2")]
        public void AuditCurrent() { try { Load();lastReport=Audit();Debug.Log(JsonUtility.ToJson(lastReport,true)); } catch(Exception e){Debug.LogError(e.Message);} }
        [ContextMenu("Export Placement Report V2")]
        public void ExportReport()
        {
            Load(); lastReport=Audit();
            string path=Path.Combine(Application.dataPath,"GeneratedLayoutsV2");Directory.CreateDirectory(path);
            File.WriteAllText(Path.Combine(path,"placement_report_v2.json"),JsonUtility.ToJson(lastReport,true));
        }
    }
}
