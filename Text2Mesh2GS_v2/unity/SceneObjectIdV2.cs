using UnityEngine;

namespace Text2Mesh2GS.V2
{
    public class SceneObjectIdV2 : MonoBehaviour
    {
        public string objectId;
        [Tooltip("The model's actual local forward axis. Normalised automatically.")]
        public Vector3 forwardAxis = Vector3.forward;
    }
}
