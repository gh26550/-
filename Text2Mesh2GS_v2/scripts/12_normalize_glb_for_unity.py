import argparse
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args():
    if "--" in sys.argv:
        script_args = sys.argv[sys.argv.index("--") + 1:]
    else:
        script_args = []

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True, help="Input GLB path")
    parser.add_argument("--output", required=True, help="Output normalized GLB path")

    parser.add_argument(
        "--target_height",
        type=float,
        default=0.0,
        help="Target object height in meters. If 0, keep original scale."
    )

    parser.add_argument(
        "--target_max_dim",
        type=float,
        default=0.0,
        help="Target maximum dimension in meters. Used only when target_height is 0."
    )

    parser.add_argument(
        "--yaw_deg",
        type=float,
        default=0.0,
        help="Additional yaw rotation in degrees around Blender Z axis."
    )

    return parser.parse_args(script_args)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_glb(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"Input GLB not found: {path}")

    bpy.ops.import_scene.gltf(filepath=str(path))


def get_mesh_objects():
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def get_root_objects():
    return [obj for obj in bpy.context.scene.objects if obj.parent is None]


def compute_world_bbox():
    bpy.context.view_layer.update()
    mesh_objects = get_mesh_objects()

    if not mesh_objects:
        raise RuntimeError("No mesh objects found.")

    min_corner = Vector((float("inf"), float("inf"), float("inf")))
    max_corner = Vector((float("-inf"), float("-inf"), float("-inf")))

    for obj in mesh_objects:
        for corner in obj.bound_box:
            world_corner = obj.matrix_world @ Vector(corner)

            min_corner.x = min(min_corner.x, world_corner.x)
            min_corner.y = min(min_corner.y, world_corner.y)
            min_corner.z = min(min_corner.z, world_corner.z)

            max_corner.x = max(max_corner.x, world_corner.x)
            max_corner.y = max(max_corner.y, world_corner.y)
            max_corner.z = max(max_corner.z, world_corner.z)

    center = (min_corner + max_corner) * 0.5
    size = max_corner - min_corner

    return min_corner, max_corner, center, size


def apply_transform_to_roots(translation=None, scale=None, yaw_deg=0.0):
    roots = get_root_objects()

    if translation is not None:
        for obj in roots:
            obj.location += translation

    if scale is not None and scale != 1.0:
        for obj in roots:
            obj.location *= scale
            obj.scale *= scale

    if yaw_deg != 0.0:
        import math
        yaw_rad = math.radians(yaw_deg)

        from mathutils import Matrix
        rotation = Matrix.Rotation(yaw_rad, 4, "Z")
        for obj in roots:
            obj.matrix_world = rotation @ obj.matrix_world
    bpy.context.view_layer.update()


def apply_all_transforms():
    bpy.ops.object.select_all(action="DESELECT")

    for obj in get_mesh_objects():
        obj.select_set(True)
        bpy.context.view_layer.objects.active = obj

    bpy.ops.object.transform_apply(location=True, rotation=True, scale=True)


def normalize_origin_and_scale(target_height: float, target_max_dim: float, yaw_deg: float):
    print("[INFO] Original bbox")
    min_corner, max_corner, center, size = compute_world_bbox()
    print(f"  min : {tuple(min_corner)}")
    print(f"  max : {tuple(max_corner)}")
    print(f"  size: {tuple(size)}")

    # 1. X/Y方向は中心、Z方向は下端を原点へ
    # Blender: X=横, Y=奥行き, Z=高さ
    translation = Vector((-center.x, -center.y, -min_corner.z))
    apply_transform_to_roots(translation=translation)

    min_corner, max_corner, center, size = compute_world_bbox()

    # 2. スケール補正
    scale_factor = 1.0

    if target_height > 0.0:
        current_height = size.z
        if current_height <= 0.0:
            raise RuntimeError("Object height is zero.")
        scale_factor = target_height / current_height

    elif target_max_dim > 0.0:
        current_max_dim = max(size.x, size.y, size.z)
        if current_max_dim <= 0.0:
            raise RuntimeError("Object max dimension is zero.")
        scale_factor = target_max_dim / current_max_dim

    if scale_factor != 1.0:
        print(f"[INFO] Scale factor: {scale_factor}")
        apply_transform_to_roots(scale=scale_factor)

    # 3. 必要なら正面方向を補正
    if yaw_deg != 0.0:
        print(f"[INFO] Apply yaw rotation: {yaw_deg} deg")
        apply_transform_to_roots(yaw_deg=yaw_deg)

    # 4. もう一度、下端中心を原点へ合わせる
    min_corner, max_corner, center, size = compute_world_bbox()
    translation = Vector((-center.x, -center.y, -min_corner.z))
    apply_transform_to_roots(translation=translation)

    # 5. Transformをメッシュに焼き込む
    apply_all_transforms()

    print("[INFO] Normalized bbox")
    min_corner, max_corner, center, size = compute_world_bbox()
    print(f"  min : {tuple(min_corner)}")
    print(f"  max : {tuple(max_corner)}")
    print(f"  size: {tuple(size)}")
    print(f"  origin should be bottom-center: x≈0, y≈0, z=0")


def export_glb(path: Path):
    path.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.export_scene.gltf(
        filepath=str(path),
        export_format="GLB",
        export_materials="EXPORT",
        export_image_format="AUTO"
    )

    print(f"[SAVED] {path}")


def main():
    args = parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    print("=" * 60)
    print(f"[INFO] Input          : {input_path}")
    print(f"[INFO] Output         : {output_path}")
    print(f"[INFO] Target height  : {args.target_height}")
    print(f"[INFO] Target max dim : {args.target_max_dim}")
    print(f"[INFO] Yaw deg        : {args.yaw_deg}")
    print("=" * 60)

    clear_scene()
    import_glb(input_path)

    normalize_origin_and_scale(
        target_height=args.target_height,
        target_max_dim=args.target_max_dim,
        yaw_deg=args.yaw_deg
    )

    export_glb(output_path)
    print("[DONE] Normalized GLB exported.")


if __name__ == "__main__":
    main()
