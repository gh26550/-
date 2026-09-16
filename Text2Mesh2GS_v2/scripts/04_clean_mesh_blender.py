import argparse
import sys
from pathlib import Path

import bpy
import bmesh


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_glb (input_path: Path):
    if not input_path.exists():
        raise FileNotFoundError(f"Input GLB not found: {input_path}")

    bpy.ops.import_scene.gltf(filepath=str(input_path))


def clean_mesh():
    mesh_objects = [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]

    if len(mesh_objects) == 0:
        raise RuntimeError("No mesh objects found.")

    for obj in mesh_objects:
        print(f"[INFO] Cleaning mesh: {obj.name}")

        bpy.ops.object.select_all(action="DESELECT")
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

        # 位置・回転・スケールを適用
        bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)

        # bmeshで重複頂点削除
        mesh = obj.data
        bm = bmesh.new()
        bm.from_mesh(mesh)

        before_verts = len(bm.verts)
        before_faces = len(bm.faces)

        bmesh.ops.remove_doubles(
            bm,
            verts=bm.verts,
            dist=0.0001
        )

        # 法線を外向きに揃える
        bmesh.ops.recalc_face_normals(
            bm,
            faces=bm.faces
        )

        bm.to_mesh(mesh)
        bm.free()

        mesh.update()

        after_verts = len(mesh.vertices)
        after_faces = len(mesh.polygons)

        print(f"[INFO] Vertices: {before_verts} -> {after_verts}")
        print(f"[INFO] Faces   : {before_faces} -> {after_faces}")


def export_glb(output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.export_scene.gltf(
        filepath=str(output_path),
        export_format="GLB"
    )


def parse_args():
    if "--" in sys.argv:
        script_args = sys.argv[sys.argv.index("--") + 1:]
    else:
        script_args = []

    parser = argparse.ArgumentParser()
    parser.add_argument("--input", required=True, help="Input GLB path")
    parser.add_argument("--output", required=True, help="Output cleaned GLB path")

    return parser.parse_args(script_args)


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_path = Path(args.output)

    print(f"[INFO] Input : {input_path}")
    print(f"[INFO] Output: {output_path}")

    clear_scene()
    import_glb(input_path)
    clean_mesh()
    export_glb(output_path)

    print(f"[DONE] Cleaned mesh saved: {output_path}")


if __name__ == "__main__":
    main()