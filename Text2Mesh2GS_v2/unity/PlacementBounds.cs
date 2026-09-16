using UnityEngine;

namespace Text2Mesh2GS.V2
{
    // Explicit adapter for renderers (e.g. procedural Gaussian renderers) without Unity Renderer.bounds.
    public class PlacementBounds : MonoBehaviour
    {
        public Bounds localBounds = new Bounds(Vector3.zero, Vector3.one);
        public bool useExplicitBounds = true;

        public static bool TryGet(Transform obj, Transform frame, out Bounds bounds)
        {
            bounds = new Bounds();
            if (obj == null) return false;
            var proxy = obj.GetComponent<PlacementBounds>();
            if (proxy != null && proxy.useExplicitBounds)
            {
                if (proxy.localBounds.size.x <= 0 || proxy.localBounds.size.y <= 0 || proxy.localBounds.size.z <= 0) return false;
                bool found = false;
                Encapsulate(proxy.localBounds, obj, frame, ref bounds, ref found);
                return found;
            }
            bool hasBounds = false;
            foreach (Renderer renderer in obj.GetComponentsInChildren<Renderer>(true))
            {
                var mesh = renderer.GetComponent<MeshFilter>();
                var skinned = renderer as SkinnedMeshRenderer;
                if (mesh != null && mesh.sharedMesh != null)
                    Encapsulate(mesh.sharedMesh.bounds, mesh.transform, frame, ref bounds, ref hasBounds);
                else if (skinned != null)
                    Encapsulate(skinned.localBounds, skinned.transform, frame, ref bounds, ref hasBounds);
                else
                    Encapsulate(renderer.bounds, null, frame, ref bounds, ref hasBounds);
            }
            return hasBounds;
        }

        private static void Encapsulate(Bounds source, Transform local, Transform frame, ref Bounds output, ref bool found)
        {
            for (int i = 0; i < 8; ++i)
            {
                Vector3 p = source.center + Vector3.Scale(source.extents,
                    new Vector3((i & 1) == 0 ? -1 : 1, (i & 2) == 0 ? -1 : 1, (i & 4) == 0 ? -1 : 1));
                if (local != null) p = local.TransformPoint(p);
                if (frame != null) p = frame.InverseTransformPoint(p);
                if (!found) { output = new Bounds(p, Vector3.zero); found = true; }
                else output.Encapsulate(p);
            }
        }
    }
}
