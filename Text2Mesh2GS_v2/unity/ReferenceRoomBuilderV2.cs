using System;
using UnityEngine;

namespace Text2Mesh2GS.V2
{
    // Optional reference geometry for a rectangular room with one back-wall opening.
    // Creates a NEW child; never deletes or replaces an existing room.
    public class ReferenceRoomBuilderV2 : MonoBehaviour
    {
        public TextAsset layoutJson;
        public Transform roomFrame;
        public float wallThickness=.1f;
        public Vector2 windowSize=new Vector2(3.2f,2.2f);
        public Vector2 windowCentre=new Vector2(0,2);
        public Transform generatedRoot;
        Transform Piece(Transform parent,string id,Vector3 position,Vector3 size)
        {
            var cube=GameObject.CreatePrimitive(PrimitiveType.Cube);cube.name=id;
            cube.transform.SetParent(parent,false);cube.transform.localPosition=position;cube.transform.localScale=size;
            return cube.transform;
        }
        Transform Anchor(Transform parent,string id)
        {
            var go=new GameObject(id);go.transform.SetParent(parent,false);go.AddComponent<SceneObjectIdV2>().objectId=id;return go.transform;
        }
        [ContextMenu("Build New Reference Room V2")]
        public void Build()
        {
            if(generatedRoot!=null)throw new InvalidOperationException("Room already generated; use a separate builder for another room.");
            if(roomFrame==null||layoutJson==null)throw new InvalidOperationException("Assign roomFrame and layout JSON.");
            var layout=JsonUtility.FromJson<LayoutData>(layoutJson.text);
            if(layout==null||layout.room==null||layout.room.size==null||layout.room.size.Length!=3)throw new InvalidOperationException("Invalid room size.");
            float w=layout.room.size[0],h=layout.room.size[1],d=layout.room.size[2],t=wallThickness;
            float l=windowCentre.x-windowSize.x/2,r=windowCentre.x+windowSize.x/2,b=windowCentre.y-windowSize.y/2,u=windowCentre.y+windowSize.y/2;
            if(t<=0||l<=-w/2||r>=w/2||b<=0||u>=h||windowSize.x<=0||windowSize.y<=0)throw new InvalidOperationException("Opening must be inside back wall.");
            generatedRoot=Anchor(roomFrame,"ReferenceRoomV2");
            Piece(generatedRoot,"floor_01",new Vector3(0,-t/2,0),new Vector3(w,t,d));
            Piece(generatedRoot,"ceiling_01",new Vector3(0,h+t/2,0),new Vector3(w,t,d));
            var left=Anchor(generatedRoot,"wall_left_01");left.localPosition=new Vector3(-w/2-t/2,h/2,0);left.localRotation=Quaternion.Euler(0,-90,0);
            Piece(left,"wall_left_piece",Vector3.zero,new Vector3(d,h,t));
            var right=Anchor(generatedRoot,"wall_right_01");right.localPosition=new Vector3(w/2+t/2,h/2,0);right.localRotation=Quaternion.Euler(0,90,0);
            Piece(right,"wall_right_piece",Vector3.zero,new Vector3(d,h,t));
            var back=Anchor(generatedRoot,"wall_back_01");back.localPosition=new Vector3(0,0,d/2+t/2);
            Piece(back,"wall_back_left",new Vector3((-w/2+l)/2,h/2,0),new Vector3(l+w/2,h,t));
            Piece(back,"wall_back_right",new Vector3((r+w/2)/2,h/2,0),new Vector3(w/2-r,h,t));
            Piece(back,"wall_back_bottom",new Vector3(windowCentre.x,b/2,0),new Vector3(windowSize.x,b,t));
            Piece(back,"wall_back_top",new Vector3(windowCentre.x,(u+h)/2,0),new Vector3(windowSize.x,h-u,t));
            var window=Anchor(generatedRoot,"window_frame_01");window.localPosition=new Vector3(windowCentre.x,windowCentre.y,d/2);
            Piece(window,"Frame_Left",new Vector3(-windowSize.x/2-t/2,0,0),new Vector3(t,windowSize.y+2*t,t));
            Piece(window,"Frame_Right",new Vector3(windowSize.x/2+t/2,0,0),new Vector3(t,windowSize.y+2*t,t));
            Piece(window,"Frame_Bottom",new Vector3(0,-windowSize.y/2-t/2,0),new Vector3(windowSize.x,t,t));
            Piece(window,"Frame_Top",new Vector3(0,windowSize.y/2+t/2,0),new Vector3(windowSize.x,t,t));
        }
    }
}
