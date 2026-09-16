import argparse
import sys
from pathlib import Path

import bpy


def parse_args():
    if "--" in sys.argv:
        script_args = sys.argv[sys.argv.index("--") + 1:]
    else:
        script_args = []

    parser = argparse.ArgumentParser()

    parser.add_argument("--image", required=True, help="Input background image path")
    parser.add_argument("--output", required=True, help="Output GLB path")
    parser.add_argument("--width", type=float, default=3.6, help="Plate width in Blender units")
    parser.add_argument("--height", type=float, default=2.4, help="Plate height in Blender units")
    parser.add_argument("--name", type=str, default="background_plate")

    return parser.parse_args(script_args)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def create_background_plate(image_path: Path, width: float, height: float, name: str):
    if not image_path.exists():
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Blender座標:
    # X = 横方向
    # Y = 奥行き方向
    # Z = 高さ方向
    #
    # 板は X-Z 平面上に作る。
    # y=0 に置き、正面から見る背景板として使う。
    verts = [
        (-width / 2.0, 0.0, 0.0),
        ( width / 2.0, 0.0, 0.0),
        ( width / 2.0, 0.0, height),
        (-width / 2.0, 0.0, height),
    ]

    faces = [
        (0, 1, 2, 3)
    ]

    mesh = bpy.data.meshes.new(name + "_mesh")
    mesh.from_pydata(verts, [], faces)
    mesh.update()

    obj = bpy.data.objects.new(name, mesh)
    bpy.context.collection.objects.link(obj)

    # UVを設定する
    uv_layer = mesh.uv_layers.new(name="UVMap")
    uv_coords = [
        (0.0, 0.0),
        (1.0, 0.0),
        (1.0, 1.0),
        (0.0, 1.0),
    ]

    for poly in mesh.polygons:
        for loop_index, uv in zip(poly.loop_indices, uv_coords):
            uv_layer.data[loop_index].uv = uv

    # 画像テクスチャ付きマテリアルを作る
    mat = bpy.data.materials.new(name + "_material")
    mat.use_nodes = True
    mat.use_backface_culling = False

    nodes = mat.node_tree.nodes
    links = mat.node_tree.links

    bsdf = nodes.get("Principled BSDF")
    tex_node = nodes.new(type="ShaderNodeTexImage")
    tex_node.image = bpy.data.images.load(str(image_path.resolve()))

    if bsdf is None:
        raise RuntimeError("Principled BSDF node not found.")

    links.new(tex_node.outputs["Color"], bsdf.inputs["Base Color"])

    # PNGにalphaがある場合にも対応
    if "Alpha" in tex_node.outputs and "Alpha" in bsdf.inputs:
        links.new(tex_node.outputs["Alpha"], bsdf.inputs["Alpha"])
        mat.blend_method = "BLEND"
        mat.show_transparent_back = True

    obj.data.materials.append(mat)

    return obj


def setup_origin_and_view(obj):
    # 原点は板の下中央に近い状態。
    # 05_render_multiview_blender_depth.py 側で recenter_scene_to_origin() が呼ばれるため、
    # ここでは無理に中心合わせしない。
    bpy.context.view_layer.objects.active = obj
    obj.select_set(True)


def export_glb(output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.export_scene.gltf(
        filepath=str(output_path),
        export_format="GLB"
    )

    print(f"[SAVED] {output_path}")


def main():
    args = parse_args()

    image_path = Path(args.image)
    output_path = Path(args.output)

    print(f"[INFO] Image : {image_path}")
    print(f"[INFO] Output: {output_path}")
    print(f"[INFO] Size  : width={args.width}, height={args.height}")

    clear_scene()

    obj = create_background_plate(
        image_path=image_path,
        width=args.width,
        height=args.height,
        name=args.name,
    )

    setup_origin_and_view(obj)
    export_glb(output_path)

    print("[DONE] Background plate GLB created.")


if __name__ == "__main__":
    main()