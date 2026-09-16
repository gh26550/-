using System;

namespace Text2Mesh2GS.V2
{
    // No Unity dependency: the same numerical rules can be regression-tested off-line.
    public static class PlacementMath
    {
        public static double Satisfaction(double error, double tolerance, double halfScoreExcess)
        {
            if (double.IsNaN(error) || double.IsInfinity(error)) return 0;
            double excess = Math.Max(0, error - tolerance);
            return 1.0 / (1.0 + excess / Math.Max(1e-6, halfScoreExcess));
        }
        public static double TargetSize(double opening, bool fitInside, double padding, double margin)
        {
            return opening * (fitInside ? Math.Max(.01, 1 - Math.Max(0, padding)) : 1 + Math.Max(0, margin));
        }
        public static bool Better(int invalid, double violation, double score,
                                  int otherInvalid, double otherViolation, double otherScore)
        {
            if (invalid != otherInvalid) return invalid < otherInvalid;
            if (Math.Abs(violation - otherViolation) > 1e-7) return violation < otherViolation;
            return score > otherScore + 1e-6;
        }
    }
}
