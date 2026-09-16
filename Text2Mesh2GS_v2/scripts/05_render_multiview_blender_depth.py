import argparse
import json
import math
import sys
from pathlib import Path
import numpy as np
import bpy
from mathutils import Vector


def parse_args():
    if "--" in sys.argv:
        script_args = sys.argv[sys.argv.index("--") + 1:]
    else:
        script_args = []

    parser = argparse.ArgumentParser()

    parser.add_argument("--input", required=True, help="Input cleaned GLB path")
    parser.add_argument("--output_dir", required=True, help="Output directory for multiview renders")
    parser.add_argument("--resolution", type=int, default=1024, help="Render resolution square")
    parser.add_argument("--num_azimuth_views", type=int, default=24, help="Number of azimuth views")
    parser.add_argument("--elevations", type=str, default="0,20", help="Comma-separated elevation angles in degrees")
    parser.add_argument("--camera_type", type=str, default="perspective", choices=["perspective", "orthographic"])
    parser.add_argument("--background", type=str, default="transparent", choices=["white", "transparent"])
    parser.add_argument("--save_camera_params", action="store_true", help="Save camera params JSON")
    parser.add_argument("--save_depth", action="store_true", help="Save Z-depth maps as OpenEXR")
    parser.add_argument(
        "--camera_margin",
        type=float,
        default=1.15,
        help="Smaller value makes object larger in image. Recommended: 1.10-1.25",
    )
    parser.add_argument(
        "--save_aligned_glb",
        action="store_true",
        help="Save recentered/aligned GLB used for rendering",
    )
    parser.add_argument(
        "--aligned_glb_path",
        type=str,
        default="",
        help="Optional output path for aligned GLB",
    )

    return parser.parse_args(script_args)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete()


def import_glb(input_path: Path):
    if not input_path.exists():
        raise FileNotFoundError(f"Input GLB not found: {input_path}")
    bpy.ops.import_scene.gltf(filepath=str(input_path))


def get_mesh_objects():
    return [obj for obj in bpy.context.scene.objects if obj.type == "MESH"]


def get_root_objects():
    return [obj for obj in bpy.context.scene.objects if obj.parent is None]


def compute_bbox(mesh_objects):
    if len(mesh_objects) == 0:
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
    return {"min": min_corner, "max": max_corner, "center": center, "size": size}


def recenter_scene_to_origin():
    mesh_objects = get_mesh_objects()
    bbox = compute_bbox(mesh_objects)
    center = bbox["center"]
    min_z = bbox["min"].z
    translation = Vector((-center.x, -center.y, -min_z))

    for obj in get_root_objects():
        obj.location += translation

    return compute_bbox(get_mesh_objects())


def setup_world(background_mode="transparent"):
    scene = bpy.context.scene
    world = bpy.data.worlds.get("World") or bpy.data.worlds.new("World")
    scene.world = world
    world.use_nodes = True

    bg_node = world.node_tree.nodes.get("Background")
    if bg_node is None:
        bg_node = world.node_tree.nodes.new(type="ShaderNodeBackground")

    bg_node.inputs[0].default_value = (1.0, 1.0, 1.0, 1.0)
    bg_node.inputs[1].default_value = 1.0

    scene.render.film_transparent = background_mode == "transparent"


def setup_render_settings(resolution=1024):
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.cycles.samples = 32
    scene.cycles.use_adaptive_sampling = True

    scene.render.resolution_x = resolution
    scene.render.resolution_y = resolution
    scene.render.resolution_percentage = 100

    scene.render.image_settings.file_format = "PNG"
    scene.render.image_settings.color_mode = "RGBA"
    scene.render.image_settings.color_depth = "8"

    # Z pass for depth output.
    scene.view_layers[0].use_pass_z = True

    scene.view_settings.view_transform = "Standard"
    scene.view_settings.look = "None"
    scene.view_settings.exposure = 0.0
    scene.view_settings.gamma = 1.0


def setup_depth_compositor(depth_dir: Path):
    """Write the Z pass through a compositor File Output node as 32-bit EXR."""
    scene = bpy.context.scene
    scene.use_nodes = True
    tree = scene.node_tree

    for node in list(tree.nodes):
        tree.nodes.remove(node)

    render_layers = tree.nodes.new(type="CompositorNodeRLayers")
    depth_output = tree.nodes.new(type="CompositorNodeOutputFile")
    depth_output.label = "DepthOutput"
    depth_output.base_path = str(depth_dir)
    depth_output.file_slots[0].path = "depth_"
    depth_output.format.file_format = "OPEN_EXR_MULTILAYER"
    depth_output.format.color_depth = "32"

    tree.links.new(render_layers.outputs["Depth"], depth_output.inputs[0])
    return depth_output


def rename_latest_depth_file(depth_dir: Path, view_index: int):
    exr_files = sorted(depth_dir.glob("depth_*.exr"))
    if len(exr_files) == 0:
        print(f"[WARN] No depth EXR found in {depth_dir}")
        return ""

    latest = exr_files[-1]
    target = depth_dir / f"{view_index:03d}.exr"
    if target.exists():
        target.unlink()
    latest.rename(target)
    print(f"[SAVED DEPTH] {target}")
    return target.name


def create_camera(camera_type, bbox, camera_margin=1.15):
    cam_data = bpy.data.cameras.new(name="RenderCamera")
    cam_obj = bpy.data.objects.new("RenderCamera", cam_data)
    bpy.context.collection.objects.link(cam_obj)
    bpy.context.scene.camera = cam_obj

    size = bbox["size"]
    max_dim = max(size.x, size.y, size.z)

    if camera_type == "orthographic":
        cam_data.type = "ORTHO"
        cam_data.ortho_scale = max_dim * camera_margin
        radius = max_dim * 4.0
    else:
        cam_data.type = "PERSP"
        cam_data.lens = 50.0
        cam_data.sensor_width = 36.0
        half_fov_y = cam_data.angle_y * 0.5
        radius = (max_dim * camera_margin) / (2.0 * math.tan(half_fov_y))
        radius = max(radius, max_dim * 1.2)

    cam_data.clip_start = 0.01
    cam_data.clip_end = 1000.0
    return cam_obj, radius


def look_at(camera_obj, target=Vector((0.0, 0.0, 0.5))):
    direction = target - camera_obj.location
    camera_obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def create_lights(bbox):
    size = bbox["size"]
    max_dim = max(size.x, size.y, size.z)
    light_dist = max_dim * 3.0 + 1.0

    lights = [
        ("KeyLight", 3000, (light_dist, -light_dist, light_dist)),
        ("FillLight", 1500, (-light_dist, -light_dist, light_dist * 0.7)),
        ("RimLight", 1000, (0.0, light_dist, light_dist)),
    ]

    for name, energy, loc in lights:
        data = bpy.data.lights.new(name=name, type="AREA")
        data.energy = energy
        data.shape = "RECTANGLE"
        data.size = max_dim * 2.0
        data.size_y = max_dim * 2.0
        obj = bpy.data.objects.new(name=name, object_data=data)
        bpy.context.collection.objects.link(obj)
        obj.location = loc
        direction = Vector((0.0, 0.0, max_dim * 0.4)) - obj.location
        obj.rotation_euler = direction.to_track_quat("-Z", "Y").to_euler()


def export_aligned_glb(output_dir: Path, aligned_glb_path: str = ""):
    if aligned_glb_path:
        out_path = Path(aligned_glb_path)
    else:
        out_path = output_dir / "aligned_mesh" / "model_aligned.glb"

    out_path.parent.mkdir(parents=True, exist_ok=True)
    bpy.ops.export_scene.gltf(filepath=str(out_path), export_format="GLB")
    print(f"[SAVED] aligned GLB: {out_path}")
    return out_path

def export_blender_mesh_npz(output_dir: Path):
    """
    Blenderワールド座標系の頂点と三角形面を直接NPZへ保存する。

    GLBを経由しないため、Blender Z-up座標とカメラ行列の座標系を
    同じ状態で07へ渡せる。
    """
    out_path = output_dir / "aligned_mesh" / "model_aligned_blender.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    depsgraph = bpy.context.evaluated_depsgraph_get()

    all_vertices = []
    all_faces = []
    vertex_offset = 0

    for obj in get_mesh_objects():
        obj_eval = obj.evaluated_get(depsgraph)
        mesh_eval = obj_eval.to_mesh()

        try:
            mesh_eval.calc_loop_triangles()

            world_matrix = obj_eval.matrix_world

            vertices = np.array(
                [
                    tuple(world_matrix @ vertex.co)
                    for vertex in mesh_eval.vertices
                ],
                dtype=np.float32,
            )

            faces = np.array(
                [
                    tuple(tri.vertices)
                    for tri in mesh_eval.loop_triangles
                ],
                dtype=np.int64,
            )

            if vertices.size == 0 or faces.size == 0:
                continue

            faces = faces + vertex_offset

            all_vertices.append(vertices)
            all_faces.append(faces)

            vertex_offset += len(vertices)

        finally:
            obj_eval.to_mesh_clear()

    if not all_vertices or not all_faces:
        raise RuntimeError("No valid mesh geometry was found for NPZ export.")

    vertices = np.concatenate(all_vertices, axis=0)
    faces = np.concatenate(all_faces, axis=0)

    np.savez_compressed(
        out_path,
        vertices=vertices,
        faces=faces,
        coordinate_system=np.array("blender_z_up"),
    )

    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bbox_size = bbox_max - bbox_min

    print(f"[SAVED] Blender mesh NPZ: {out_path}")
    print(f"[INFO] NPZ vertices     : {len(vertices)}")
    print(f"[INFO] NPZ faces        : {len(faces)}")
    print(f"[INFO] NPZ bbox min     : {bbox_min}")
    print(f"[INFO] NPZ bbox max     : {bbox_max}")
    print(f"[INFO] NPZ bbox size    : {bbox_size}")

    return out_path

def render_multiview(
    input_path: Path,
    output_dir: Path,
    resolution: int,
    num_azimuth_views: int,
    elevations_deg,
    camera_type: str,
    background: str,
    save_camera_params: bool,
    camera_margin: float,
    save_aligned_glb: bool,
    aligned_glb_path: str,
    save_depth: bool,
):
    clear_scene()
    import_glb(input_path)

    bbox = recenter_scene_to_origin()

    aligned_glb_saved_path = None
    if save_aligned_glb:
        aligned_glb_saved_path = export_aligned_glb(output_dir, aligned_glb_path)

    blender_npz_saved_path = export_blender_mesh_npz(output_dir)

    setup_render_settings(resolution)
    setup_world(background)

    images_dir = output_dir / "images"
    depths_dir = output_dir / "depths"
    cameras_dir = output_dir / "cameras"
    images_dir.mkdir(parents=True, exist_ok=True)
    cameras_dir.mkdir(parents=True, exist_ok=True)
    if save_depth:
        depths_dir.mkdir(parents=True, exist_ok=True)
        setup_depth_compositor(depths_dir)

    create_lights(bbox)
    cam_obj, radius = create_camera(camera_type, bbox, camera_margin)

    scene = bpy.context.scene
    target = Vector((0.0, 0.0, bbox["size"].z * 0.45))

    camera_records = []
    view_index = 0

    for elev_deg in elevations_deg:
        elev_rad = math.radians(elev_deg)

        for az_idx in range(num_azimuth_views):
            az_deg = az_idx * (360.0 / num_azimuth_views)
            az_rad = math.radians(az_deg)

            x = radius * math.cos(elev_rad) * math.sin(az_rad)
            y = -radius * math.cos(elev_rad) * math.cos(az_rad)
            z = radius * math.sin(elev_rad) + bbox["size"].z * 0.45

            cam_obj.location = (x, y, z)
            look_at(cam_obj, target)

            filename = f"{view_index:03d}.png"
            out_path = images_dir / filename
            scene.render.filepath = str(out_path)

            # Keep frame constant so File Output writes depth_0001.exr every time,
            # then rename it to the view index immediately after rendering.
            scene.frame_set(1)
            bpy.ops.render.render(write_still=True)
            print(f"[RENDERED] {out_path}")

            depth_filename = ""
            if save_depth:
                depth_filename = rename_latest_depth_file(depths_dir, view_index)

            record = {
                "index": view_index,
                "filename": filename,
                "depth_filename": depth_filename,
                "azimuth_deg": az_deg,
                "elevation_deg": elev_deg,
                "camera_type": camera_type,
                "location": [float(v) for v in cam_obj.location],
                "rotation_euler": [float(v) for v in cam_obj.rotation_euler],
                "matrix_world": [[float(v) for v in row] for row in cam_obj.matrix_world],
                "resolution": resolution,
            }

            if camera_type == "orthographic":
                record["ortho_scale"] = float(cam_obj.data.ortho_scale)
            else:
                record["lens_mm"] = float(cam_obj.data.lens)
                record["sensor_width"] = float(cam_obj.data.sensor_width)
                record["angle_x"] = float(cam_obj.data.angle_x)
                record["angle_y"] = float(cam_obj.data.angle_y)

            camera_records.append(record)
            view_index += 1

    if save_camera_params:
        meta = {
            "input_glb": str(input_path),
            "aligned_glb": str(aligned_glb_saved_path) if aligned_glb_saved_path is not None else "",
            "camera_type": camera_type,
            "background": background,
            "blender_mesh_npz": str(blender_npz_saved_path),
            "camera_margin": camera_margin,
            "save_depth": save_depth,
            "depth_dir": "depths" if save_depth else "",
            "num_azimuth_views": num_azimuth_views,
            "elevations_deg": elevations_deg,
            "bbox_size": [float(v) for v in bbox["size"]],
            "views": camera_records,
        }
        json_path = cameras_dir / "camera_params.json"
        with open(json_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
        print(f"[SAVED] {json_path}")


def main():
    args = parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    elevations_deg = [float(x.strip()) for x in args.elevations.split(",") if x.strip()]

    print(f"[INFO] Input             : {input_path}")
    print(f"[INFO] Output dir        : {output_dir}")
    print(f"[INFO] Resolution        : {args.resolution}")
    print(f"[INFO] Azimuth views     : {args.num_azimuth_views}")
    print(f"[INFO] Elevations        : {elevations_deg}")
    print(f"[INFO] Camera type       : {args.camera_type}")
    print(f"[INFO] Background        : {args.background}")
    print(f"[INFO] Camera margin     : {args.camera_margin}")
    print(f"[INFO] Save depth        : {args.save_depth}")
    print(f"[INFO] Save camera params: {args.save_camera_params}")
    print(f"[INFO] Save aligned GLB  : {args.save_aligned_glb}")
    print(f"[INFO] Aligned GLB path  : {args.aligned_glb_path}")

    render_multiview(
        input_path=input_path,
        output_dir=output_dir,
        resolution=args.resolution,
        num_azimuth_views=args.num_azimuth_views,
        elevations_deg=elevations_deg,
        camera_type=args.camera_type,
        background=args.background,
        save_camera_params=args.save_camera_params,
        camera_margin=args.camera_margin,
        save_aligned_glb=args.save_aligned_glb,
        aligned_glb_path=args.aligned_glb_path,
        save_depth=args.save_depth,
    )

    print("[DONE] Multiview rendering completed.")






# ============================================================
# Blender 5.1 safe depth output override v4
# Viewer Node may return 512x512 single-channel depth.
# Save depth as .npy and record actual shape.
# ============================================================
def setup_depth_compositor(depth_dir: Path):
    scene = bpy.context.scene
    depth_dir.mkdir(parents=True, exist_ok=True)

    if hasattr(scene.render, "use_compositing"):
        scene.render.use_compositing = True

    if hasattr(scene, "compositing_node_group"):
        tree = bpy.data.node_groups.new(
            name="DepthViewerCompositor",
            type="CompositorNodeTree"
        )
        scene.compositing_node_group = tree
    else:
        scene.use_nodes = True
        tree = scene.node_tree

    for node in list(tree.nodes):
        tree.nodes.remove(node)

    render_layers = tree.nodes.new(type="CompositorNodeRLayers")
    viewer = tree.nodes.new(type="CompositorNodeViewer")
    viewer.label = "DepthViewer"

    depth_socket = None
    for name in ("Depth", "Z"):
        if name in render_layers.outputs:
            depth_socket = render_layers.outputs[name]
            break

    if depth_socket is None:
        print("[DEBUG] Render Layers outputs:", [o.name for o in render_layers.outputs])
        raise RuntimeError("Render Layers node has no Depth/Z output.")

    tree.links.new(depth_socket, viewer.inputs[0])

    print("[INFO] Depth compositor uses Viewer Node. Depth will be saved as .npy")
    return viewer


def rename_latest_depth_file(depth_dir: Path, view_index: int):
    import math
    import numpy as np

    depth_dir.mkdir(parents=True, exist_ok=True)

    viewer_img = bpy.data.images.get("Viewer Node")

    if viewer_img is None:
        print("[WARN] Viewer Node image not found. Depth was not saved.")
        return ""

    pixels = np.asarray(viewer_img.pixels[:], dtype=np.float32)

    if pixels.size == 0:
        print("[WARN] Viewer Node has no pixels. Depth was not saved.")
        return ""

    # Blender 5.1 Viewer Node may return:
    # - single channel: H*W
    # - RGBA: H*W*4
    # Sometimes Viewer Node resolution is 512 even if final render is 1024.
    if pixels.size % 4 == 0:
        side_rgba = int(round(math.sqrt(pixels.size / 4)))
    else:
        side_rgba = -1

    side_gray = int(round(math.sqrt(pixels.size)))

    if side_rgba > 0 and side_rgba * side_rgba * 4 == pixels.size:
        rgba = pixels.reshape((side_rgba, side_rgba, 4))
        depth = rgba[..., 0].astype(np.float32)
        mode = "rgba"

    elif side_gray * side_gray == pixels.size:
        depth = pixels.reshape((side_gray, side_gray)).astype(np.float32)
        mode = "single"

    else:
        print("[WARN] Could not infer Viewer Node depth shape.")
        print(f"[WARN] pixels.size={pixels.size}")
        return ""

    depth[~np.isfinite(depth)] = 0.0

    target = depth_dir / f"{view_index:03d}.npy"
    np.save(target, depth)

    valid = depth[depth > 0]

    if valid.size > 0:
        print(
            f"[SAVED DEPTH] {target} "
            f"mode={mode} "
            f"shape={depth.shape} "
            f"min={float(valid.min()):.6f} "
            f"max={float(valid.max()):.6f} "
            f"mean={float(valid.mean()):.6f}"
        )
    else:
        print(f"[SAVED DEPTH] {target} mode={mode} shape={depth.shape} but all depth values are 0")

    return target.name


if __name__ == "__main__":
    main()
