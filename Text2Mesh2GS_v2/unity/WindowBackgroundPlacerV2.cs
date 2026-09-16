using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using UnityEngine;

namespace Text2Mesh2GS.V2
{
    /// <summary>One explicit window and one Gaussian. No implicit name search, renderer clipping or camera visibility claim.</summary>
    public class WindowBackgroundPlacerV2 : MonoBehaviour
    {
        public TextAsset constraintsJson;
        public string gaussianId="outside_view_01", windowId="window_frame_01", wallId="wall_back_01";
        public Transform gaussian;
        [Tooltip("Unit world scale; local +Z points outdoors, +X right, +Y up. May be rotated with the room.")]
        public Transform windowBasis;
        public Transform frameLeft, frameRight, frameTop, frameBottom;
        public bool useExplicitOpening;
        public Bounds explicitOpening = new Bounds(Vector3.zero,new Vector3(3.2f,2.2f,.1f));
        public bool fitInside=true, autoAxisCorrection=true;
        public Vector3 baseScale=Vector3.one, manualAxisCorrection;
        [Range(0,.45f)] public float insidePadding=.03f;
        [Range(0,1)] public float coverMargin=.15f;
        public float behindGap=.25f;
        public string lastStatus;
        public ConstraintResult[] lastResults;

        Bounds Opening()
        {
            if(windowBasis==null||gaussian==null)throw new InvalidOperationException("Assign windowBasis and Gaussian.");
            if((windowBasis.lossyScale-Vector3.one).sqrMagnitude>1e-6f)throw new InvalidOperationException("windowBasis must have unit world scale.");
            if(useExplicitOpening)
            {
                if(explicitOpening.size.x<=0||explicitOpening.size.y<=0)throw new InvalidOperationException("Invalid explicit opening.");
                return explicitOpening;
            }
            if(!PlacementBounds.TryGet(frameLeft,windowBasis,out var left)||!PlacementBounds.TryGet(frameRight,windowBasis,out var right)
                ||!PlacementBounds.TryGet(frameTop,windowBasis,out var top)||!PlacementBounds.TryGet(frameBottom,windowBasis,out var bottom))
                throw new InvalidOperationException("Assign the four frame parts or an explicit opening.");
            Vector3 min=new Vector3(left.max.x,bottom.max.y,Mathf.Min(left.min.z,right.min.z));
            Vector3 max=new Vector3(right.min.x,top.min.y,Mathf.Max(left.max.z,right.max.z));
            if(max.x<=min.x||max.y<=min.y)throw new InvalidOperationException("Window parts do not form an opening in windowBasis.");
            return new Bounds((min+max)/2,max-min);
        }
        ConstraintData[] Constraints()
        {
            if(constraintsJson==null)throw new InvalidOperationException("Assign schema 2.0 constraints.");
            var data=JsonUtility.FromJson<ConstraintFile>(constraintsJson.text);
            if(data==null||data.schema_version!="2.0"||data.constraints==null)throw new InvalidOperationException("Expected constraint schema 2.0.");
            return data.constraints.Where(c=>c.source==gaussianId).ToArray();
        }
        bool Has(ConstraintData[] data,string relation)
        { return data.Any(c=>c.relation==relation&&(c.target==windowId||c.target==wallId)); }
        bool Bounds(out Bounds b) { return PlacementBounds.TryGet(gaussian,windowBasis,out b); }

        [ContextMenu("Apply Window Background V2")]
        public void Apply()
        {
            Vector3 oldPos=gaussian!=null?gaussian.position:Vector3.zero;
            Vector3 oldScale=gaussian!=null?gaussian.localScale:Vector3.one;
            Quaternion oldRot=gaussian!=null?gaussian.rotation:Quaternion.identity;
            try
            {
                var data=Constraints(); var opening=Opening();
                bool behind=Has(data,"behind"), centre=Has(data,"center aligned with")||Has(data,"aligned with"),
                    parallel=Has(data,"parallel to"), size=Has(data,"cover")||Has(data,"visible through");
                if(!behind&&!centre&&!parallel&&!size) { lastStatus="no_applicable_constraints"; return; }
                if(!Bounds(out var current))throw new InvalidOperationException("Gaussian Bounds unavailable. Add PlacementBounds with measured localBounds.");
                // Scale is reset only when the explicit size operation is enabled.
                if(size)
                {
                    if(baseScale.x<=0||baseScale.y<=0||baseScale.z<=0)throw new InvalidOperationException("Base scale must be positive.");
                    gaussian.localScale=baseScale;
                }
                if(parallel)
                {
                    Quaternion basis=windowBasis.rotation*Quaternion.Euler(0,180,0);
                    var corrections=new List<Vector3>{manualAxisCorrection};
                    if(autoAxisCorrection)corrections.AddRange(new[]{Vector3.zero,new Vector3(90,0,0),new Vector3(-90,0,0),new Vector3(0,0,90),new Vector3(0,0,-90),new Vector3(180,0,0),new Vector3(0,180,0)});
                    Quaternion best=basis; float bestError=float.PositiveInfinity;
                    foreach(var correction in corrections.Distinct())
                    {
                        gaussian.rotation=basis*Quaternion.Euler(correction);
                        if(!Bounds(out var test)||test.size.x<1e-6f||test.size.y<1e-6f)continue;
                        float aspect=Mathf.Abs(Mathf.Log((test.size.x/test.size.y)/(opening.size.x/opening.size.y)));
                        float thin=test.size.z/Mathf.Max(test.size.x,test.size.y);
                        float error=aspect+thin;
                        if(error<bestError){bestError=error;best=gaussian.rotation;}
                    }
                    gaussian.rotation=best;
                }
                if(!Bounds(out current))throw new InvalidOperationException("Bounds lost after rotation.");
                if(size)
                {
                    float targetW=(float)PlacementMath.TargetSize(opening.size.x,fitInside,insidePadding,coverMargin);
                    float targetH=(float)PlacementMath.TargetSize(opening.size.y,fitInside,insidePadding,coverMargin);
                    if(current.size.x<=1e-6f||current.size.y<=1e-6f)throw new InvalidOperationException("Degenerate Gaussian Bounds.");
                    float multiplier=fitInside?Mathf.Min(targetW/current.size.x,targetH/current.size.y):Mathf.Max(targetW/current.size.x,targetH/current.size.y);
                    if(multiplier<.001f||multiplier>1000f)throw new InvalidOperationException("Scale multiplier out of range; check model units.");
                    gaussian.localScale*=multiplier;
                }
                Bounds(out current);
                Vector3 delta=Vector3.zero;
                if(centre){delta.x=opening.center.x-current.center.x;delta.y=opening.center.y-current.center.y;}
                // Behind refers to the nearest face, so the whole conservative bound is outdoors.
                if(behind)delta.z=opening.max.z+Mathf.Max(0,behindGap)-current.min.z;
                gaussian.position+=windowBasis.TransformVector(delta);
                Audit();
            }
            catch(Exception e)
            {
                if(gaussian!=null){gaussian.SetPositionAndRotation(oldPos,oldRot);gaussian.localScale=oldScale;}
                lastStatus="failed: "+e.Message; Debug.LogError(lastStatus);
            }
        }
        [ContextMenu("Audit Window Background V2")]
        public void Audit()
        {
            var opening=Opening();var data=Constraints();var rows=new List<ConstraintResult>();
            bool hasBounds=Bounds(out var b);
            foreach(var c in data)
            {
                var r=new ConstraintResult{source=c.source,target=c.target,relation=c.relation,weight=c.weight,status="unsupported"};
                if(!hasBounds){r.status="missing_bounds";rows.Add(r);continue;}
                if(c.target!=windowId&&c.target!=wallId){r.status="missing_target";rows.Add(r);continue;}
                switch(c.relation)
                {
                    case "behind": r.measured=Mathf.Max(0,opening.max.z+behindGap-b.min.z); r.tolerance=.001f;break;
                    case "center aligned with":case "aligned with": r.measured=Vector2.Distance(new Vector2(b.center.x,b.center.y),new Vector2(opening.center.x,opening.center.y));r.tolerance=.01f;break;
                    case "parallel to": r.measured=b.size.z/Mathf.Max(.0001f,Mathf.Max(b.size.x,b.size.y));r.tolerance=.1f;break;
                    case "cover":
                        r.measured=Mathf.Max(0,b.min.x-opening.min.x)+Mathf.Max(0,opening.max.x-b.max.x)+Mathf.Max(0,b.min.y-opening.min.y)+Mathf.Max(0,opening.max.y-b.max.y);
                        r.tolerance=.001f;break;
                    case "visible through":r.status="requires_camera_validation";rows.Add(r);continue;
                    default:rows.Add(r);continue;
                }
                r.status=r.measured<=r.tolerance?"ok":"unmet";
                r.satisfaction=(float)PlacementMath.Satisfaction(r.measured,r.tolerance,.1);
                r.contribution=100*r.weight*r.satisfaction;rows.Add(r);
            }
            lastResults=rows.ToArray();
            bool inside=hasBounds&&b.min.x>=opening.min.x-.001f&&b.max.x<=opening.max.x+.001f&&b.min.y>=opening.min.y-.001f&&b.max.y<=opening.max.y+.001f;
            lastStatus=rows.Any(r=>r.status=="unmet"||r.status=="unsupported"||r.status.StartsWith("missing"))||(fitInside&&!inside)?"incomplete":"geometry_ready";
            if(rows.Any(r=>r.status=="requires_camera_validation"))lastStatus+="; camera validation pending";
            Debug.Log("[Window V2] "+lastStatus);
        }
        [Serializable] class WindowReport { public string status;public bool fit_inside;public ConstraintResult[] constraints; }
        [ContextMenu("Export Window Report V2")]
        public void ExportReport()
        {
            Audit();string dir=Path.Combine(Application.dataPath,"GeneratedLayoutsV2");Directory.CreateDirectory(dir);
            File.WriteAllText(Path.Combine(dir,"window_report_v2.json"),JsonUtility.ToJson(new WindowReport{status=lastStatus,fit_inside=fitInside,constraints=lastResults},true));
        }
    }
}
