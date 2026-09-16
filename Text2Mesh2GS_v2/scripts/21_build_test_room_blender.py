#!/usr/bin/env python3
"""
Build a small test room as a Mesh scene in Blender.

Purpose
-------
This script creates the room architecture (floor, walls, ceiling) as Mesh and
imports available furniture/object GLBs. It does NOT create Gaussian background
data. Gaussian should remain limited to a separate non-interactable background
asset, such as an exterior view outside a window.

Run with Blender:
    blender --background --python scripts/21_build_test_room_blender.py -- \
      --chair outputs/meshes/001_sample_object/model_chair.glb \
      --table outputs/meshes/001_sample_object/model_table.glb \
      --bookshelf outputs/images_preprocessed/004_bookshelf/sample_2026-06-02T120501.296.glb \
      --plant outputs/images_preprocessed/005_potted_plant/sample_2026-06-02T120622.131.glb \
      --vase outputs/images_preprocessed/003_cramic_vase/sample_2026-06-02T114620.292.glb \
      --output outputs/rooms/room_001/room_mesh.glb \
      --blend_output outputs/rooms/room_001/room_001.blend

All paths are resolved relative to the current working directory.
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import bpy
from mathutils import Vector


def parse_args() -> argparse.Namespace:
    argv = sys.argv
    argv = argv[argv.index("--") + 1:] if "--" in argv else []

    parser = argparse.ArgumentParser()
    parser.add_argument("--chair", default="")
    parser.add_argument("--table", default="")
    parser.add_argument("--bookshelf", default="")
    parser.add_argument("--plant", default="")
    parser.add_argument("--vase", default="")
    parser.add_argument("--output", required=True)
    parser.add_argument("--blend_output", default="")
    parser.add_argument("--room_width", type=float, default=6.0)
    parser.add_argument("--room_depth", type=float, default=6.0)
    parser.add_argument("--room_height", type=float, default=3.0)
    parser.add_argument("--wall_thickness", type=float, default=0.12)
    return parser.parse_args(argv)


def clear_scene() -> None:
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)

    for datablocks in (
        bpy.data.meshes,
        bpy.data.curves,
        bpy.data.materials,
        bpy.data.cameras,
        bpy.data.lights,
    ):
        # Do not remove materials here because imported objects may need them later.
        if datablocks is bpy.data.materials:
            continue
        for block in list(datablocks):
            if block.users == 0:
                datablocks.remove(block)


def make_material(name: str, rgba: tuple[float, float, float, float]):
    material = bpy.data.materials.new(name=name)
    material.use_nodes = True

    bsdf = material.node_tree.nodes.get("Principled BSDF")
    if bsdf is not None:
        bsdf.inputs["Base Color"].default_value = rgba
        bsdf.inputs["Roughness"].default_value = 0.7

    return material


def add_box(
    name: str,
    location: tuple[float, float, float],
    scale: tuple[float, float, float],
    material,
):
    bpy.ops.mesh.primitive_cube_add(location=location)
    obj = bpy.context.active_object
    obj.name = name
    obj.scale = scale
    bpy.ops.object.transform_apply(location=False, rotation=False, scale=True)

    if material is not None:
        obj.data.materials.append(material)

    return obj


def create_room(
    width: float,
    depth: float,
    height: float,
    thickness: float,
):
    floor_mat = make_material("FloorMaterial", (0.35, 0.22, 0.12, 1.0))
    wall_mat = make_material("WallMaterial", (0.78, 0.78, 0.74, 1.0))
    ceiling_mat = make_material("CeilingMaterial", (0.9, 0.9, 0.9, 1.0))

    add_box(
        "Floor",
        (0.0, 0.0, -thickness / 2.0),
        (width / 2.0, depth / 2.0, thickness / 2.0),
        floor_mat,
    )

    add_box(
        "Ceiling",
        (0.0, 0.0, height + thickness / 2.0),
        (width / 2.0, depth / 2.0, thickness / 2.0),
        ceiling_mat,
    )

    add_box(
        "Wall_Back",
        (0.0, depth / 2.0 + thickness / 2.0, height / 2.0),
        (width / 2.0, thickness / 2.0, height / 2.0),
        wall_mat,
    )

    add_box(
        "Wall_Left",
        (-width / 2.0 - thickness / 2.0, 0.0, height / 2.0),
        (thickness / 2.0, depth / 2.0, height / 2.0),
        wall_mat,
    )

    add_box(
        "Wall_Right",
        (width / 2.0 + thickness / 2.0, 0.0, height / 2.0),
        (thickness / 2.0, depth / 2.0, height / 2.0),
        wall_mat,
    )

    # Front wall with a central opening so the room can be inspected easily.
    opening_width = 2.4
    side_width = (width - opening_width) / 2.0

    add_box(
        "Wall_Front_Left",
        (-(opening_width / 2.0 + side_width / 2.0),
         -depth / 2.0 - thickness / 2.0,
         height / 2.0),
        (side_width / 2.0, thickness / 2.0, height / 2.0),
        wall_mat,
    )

    add_box(
        "Wall_Front_Right",
        ((opening_width / 2.0 + side_width / 2.0),
         -depth / 2.0 - thickness / 2.0,
         height / 2.0),
        (side_width / 2.0, thickness / 2.0, height / 2.0),
        wall_mat,
    )


def imported_objects_before() -> set[str]:
    return {obj.name for obj in bpy.context.scene.objects}


def import_glb(path: Path) -> list[bpy.types.Object]:
    before = imported_objects_before()
    bpy.ops.import_scene.gltf(filepath=str(path))
    return [
        obj for obj in bpy.context.scene.objects
        if obj.name not in before
    ]


def mesh_bounds(objects: list[bpy.types.Object]):
    corners = []

    for obj in objects:
        if obj.type != "MESH":
            continue
        for corner in obj.bound_box:
            corners.append(obj.matrix_world @ Vector(corner))

    if not corners:
        return None

    min_corner = Vector((
        min(v.x for v in corners),
        min(v.y for v in corners),
        min(v.z for v in corners),
    ))
    max_corner = Vector((
        max(v.x for v in corners),
        max(v.y for v in corners),
        max(v.z for v in corners),
    ))
    return min_corner, max_corner


def create_parent_empty(name: str):
    empty = bpy.data.objects.new(name, None)
    bpy.context.collection.objects.link(empty)
    return empty


def normalize_and_place(
    objects: list[bpy.types.Object],
    name: str,
    target_position: tuple[float, float, float],
    target_height: float,
    yaw_degrees: float = 0.0,
):
    bounds = mesh_bounds(objects)
    if bounds is None:
        raise RuntimeError(f"No mesh was imported for {name}")

    min_corner, max_corner = bounds
    current_height = max_corner.z - min_corner.z

    parent = create_parent_empty(name)

    for obj in objects:
        if obj.parent is None:
            obj.parent = parent

    if current_height <= 1e-6:
        scale = 1.0
    else:
        scale = target_height / current_height

    parent.scale = (scale, scale, scale)
    bpy.context.view_layer.update()

    bounds = mesh_bounds(objects)
    if bounds is None:
        raise RuntimeError(f"Failed to compute scaled bounds for {name}")

    min_corner, max_corner = bounds
    center_xy = Vector((
        (min_corner.x + max_corner.x) / 2.0,
        (min_corner.y + max_corner.y) / 2.0,
        0.0,
    ))

    parent.location.x += target_position[0] - center_xy.x
    parent.location.y += target_position[1] - center_xy.y
    parent.location.z += target_position[2] - min_corner.z
    parent.rotation_euler.z = math.radians(yaw_degrees)

    bpy.context.view_layer.update()
    return parent


def add_asset(
    label: str,
    path_text: str,
    position: tuple[float, float, float],
    target_height: float,
    yaw: float = 0.0,
):
    if not path_text:
        print(f"[SKIP] {label}: no input path")
        return None

    path = Path(path_text).expanduser().resolve()
    if not path.exists():
        print(f"[SKIP] {label}: file not found: {path}")
        return None

    print(f"[IMPORT] {label}: {path}")
    objects = import_glb(path)
    parent = normalize_and_place(
        objects,
        label,
        position,
        target_height,
        yaw,
    )
    return parent


def add_lighting():
    bpy.ops.object.light_add(
        type="AREA",
        location=(0.0, 0.0, 2.7),
    )
    key = bpy.context.active_object
    key.name = "RoomAreaLight"
    key.data.energy = 1100.0
    key.data.shape = "DISK"
    key.data.size = 4.0

    bpy.ops.object.light_add(
        type="SUN",
        location=(0.0, 0.0, 2.5),
    )
    sun = bpy.context.active_object
    sun.name = "Sun"
    sun.rotation_euler = (
        math.radians(25.0),
        math.radians(-20.0),
        math.radians(25.0),
    )
    sun.data.energy = 1.5


def export_scene(output_path: Path):
    output_path.parent.mkdir(parents=True, exist_ok=True)

    bpy.ops.object.select_all(action="DESELECT")
    for obj in bpy.context.scene.objects:
        if obj.type in {"MESH", "EMPTY"}:
            obj.select_set(True)

    bpy.ops.export_scene.gltf(
        filepath=str(output_path),
        export_format="GLB",
        use_selection=True,
        export_apply=True,
        export_yup=True,
        export_materials="EXPORT",
    )


def main():
    args = parse_args()
    clear_scene()

    create_room(
        args.room_width,
        args.room_depth,
        args.room_height,
        args.wall_thickness,
    )

    add_asset(
        "Chair",
        args.chair,
        (-1.0, 0.5, 0.0),
        1.0,
        -20.0,
    )
    add_asset(
        "Table",
        args.table,
        (0.4, 0.4, 0.0),
        0.8,
        0.0,
    )
    add_asset(
        "Bookshelf",
        args.bookshelf,
        (2.15, 2.3, 0.0),
        2.0,
        180.0,
    )
    add_asset(
        "Plant",
        args.plant,
        (-2.0, 2.2, 0.0),
        1.25,
        0.0,
    )
    add_asset(
        "Vase",
        args.vase,
        (0.4, 0.4, 0.82),
        0.35,
        0.0,
    )

    add_lighting()

    output_path = Path(args.output).expanduser().resolve()
    export_scene(output_path)
    print(f"[SAVED] GLB: {output_path}")

    if args.blend_output:
        blend_path = Path(args.blend_output).expanduser().resolve()
        blend_path.parent.mkdir(parents=True, exist_ok=True)
        bpy.ops.wm.save_as_mainfile(filepath=str(blend_path))
        print(f"[SAVED] BLEND: {blend_path}")


if __name__ == "__main__":
    main()
