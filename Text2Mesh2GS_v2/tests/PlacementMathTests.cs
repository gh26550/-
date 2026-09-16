using System;
using Text2Mesh2GS.V2;
public class PlacementMathTests
{
    static void Check(bool condition,string name) { if(!condition)throw new Exception(name); }
    public static void Main()
    {
        Check(PlacementMath.Satisfaction(.1,.1,1)==1,"boundary full score");
        Check(PlacementMath.Satisfaction(double.PositiveInfinity,.1,1)==0,"missing not full score");
        Check(PlacementMath.Satisfaction(180,10,35)<.2,"opposite facing low score");
        Check(Math.Abs(PlacementMath.TargetSize(2,true,.03,.15)-1.94)<1e-9,"fit ignores cover margin");
        Check(Math.Abs(PlacementMath.TargetSize(2,false,.03,.15)-2.3)<1e-9,"cover ignores inside padding");
        Check(PlacementMath.Better(0,0,10,1,.1,1000),"feasibility before semantic score");
        Check(!PlacementMath.Better(1,.1,1000,0,0,10),"infeasible cannot beat feasible");
        Console.WriteLine("7 numerical regression checks passed.");
    }
}
