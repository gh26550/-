#if UNITY_EDITOR
using System;
using UnityEditor;
using UnityEngine;
using Text2Mesh2GS.V2;

// Copy into an Editor folder in a disposable Unity project with unity/*.cs.
public static class PlacementV2Smoke
{
    static void Check(bool value,string message) { if(!value)throw new Exception(message); }
    static GameObject Cube(string name,Transform parent,Vector3 size)
    {
        var go=GameObject.CreatePrimitive(PrimitiveType.Cube);go.name=name;go.transform.SetParent(parent,false);go.transform.localScale=size;
        var marker=go.AddComponent<SceneObjectIdV2>();marker.objectId=name;
        return go;
    }
    public static void Run()
    {
        try
        {
            var room=new GameObject("room");room.transform.SetPositionAndRotation(new Vector3(5,2,-3),Quaternion.Euler(0,37,0));
            var movable=new GameObject("movable");movable.transform.SetParent(room.transform,false);
            var refs=new GameObject("references");refs.transform.SetParent(room.transform,false);
            var table=Cube("table",movable.transform,Vector3.one);
            var vase=Cube("vase",movable.transform,Vector3.one*.2f);
            var manager=new GameObject("manager");var solver=manager.AddComponent<SemanticScenePlacerV2>();
            solver.roomFrame=room.transform;solver.movableRoot=movable.transform;solver.referenceRoot=refs.transform;
            solver.candidateLimit=16;solver.refinementSweeps=1;
            solver.layoutJson=new TextAsset("{\"schema_version\":\"2.0\",\"scene_id\":\"test\",\"room\":{\"size\":[6,4,6]},\"unity_objects\":[{\"id\":\"table\",\"movable\":true,\"representation\":\"mesh\"},{\"id\":\"vase\",\"movable\":true,\"representation\":\"mesh\"}]}");
            solver.constraintsJson=new TextAsset("{\"schema_version\":\"2.0\",\"scene_id\":\"test\",\"constraints\":[{\"source\":\"table\",\"target\":\"room\",\"relation\":\"center of room\",\"weight\":1},{\"source\":\"vase\",\"target\":\"table\",\"relation\":\"on\",\"weight\":1}]}");
            solver.ApplyLayout();
            Check(solver.lastReport!=null&&solver.lastReport.status=="success","Support placement with rotated/translated room failed");
            Vector3 first=table.transform.position;
            table.transform.position+=Vector3.one*7;vase.transform.position-=Vector3.one*4;
            solver.ApplyLayout();Check(Vector3.Distance(first,table.transform.position)<1e-4f,"Deterministic start depends on prior positions");
            UnityEngine.Object.DestroyImmediate(vase);
            solver.AuditCurrent();Check(solver.lastReport.coverage_100==50,"Missing source omitted from coverage");
            Check(solver.lastReport.score_100==50&&!solver.lastReport.physically_feasible,"Missing source score/status");
            var gaussian=Cube("background",room.transform,new Vector3(3,2,.1f));
            var window=manager.AddComponent<WindowBackgroundPlacerV2>();window.gaussian=gaussian.transform;window.windowBasis=room.transform;
            window.useExplicitOpening=true;window.explicitOpening=new Bounds(new Vector3(0,2,3),new Vector3(2,1,.1f));
            window.constraintsJson=new TextAsset("{\"schema_version\":\"2.0\",\"constraints\":[{\"source\":\"outside_view_01\",\"target\":\"window_frame_01\",\"relation\":\"behind\",\"weight\":1},{\"source\":\"outside_view_01\",\"target\":\"window_frame_01\",\"relation\":\"center aligned with\",\"weight\":1},{\"source\":\"outside_view_01\",\"target\":\"window_frame_01\",\"relation\":\"visible through\",\"weight\":1}]}");
            window.Apply();
            PlacementBounds.TryGet(gaussian.transform,room.transform,out var bounds);
            Check(bounds.size.x<=2&&bounds.size.y<=1,"Fit inside exceeded opening");
            Check(bounds.min.z>=3.05f+.25f-.001f,"Near face is not outside window");
            Check(window.lastStatus.Contains("camera validation pending"),"Visibility falsely reported complete");
            Vector3 scale=gaussian.transform.localScale;
            window.constraintsJson=new TextAsset("{\"schema_version\":\"2.0\",\"constraints\":[]}");window.Apply();
            Check(gaussian.transform.localScale==scale,"No-constraint apply changed scale");
            var builder=manager.AddComponent<ReferenceRoomBuilderV2>();builder.roomFrame=room.transform;builder.layoutJson=solver.layoutJson;
            builder.Build();
            var left=builder.generatedRoot.Find("wall_left_01");var right=builder.generatedRoot.Find("wall_right_01");
            Check(left.localPosition.x<0&&right.localPosition.x>0,"Reference walls not separated");
            UnityEngine.Object.DestroyImmediate(room);UnityEngine.Object.DestroyImmediate(manager);
            Debug.Log("V2_SMOKE_PASS");EditorApplication.Exit(0);
        }
        catch(Exception e){Debug.LogException(e);EditorApplication.Exit(1);}
    }
}
#endif
