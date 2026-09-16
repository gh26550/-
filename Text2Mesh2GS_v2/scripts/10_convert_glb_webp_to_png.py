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
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--texture_dir",
        default="",
        help="Temporary PNG texture output directory",
    )
    return parser.parse_args(script_args)


def clear_scene():
    bpy.ops.object.select_all(action="SELECT")
    bpy.ops.object.delete(use_global=False)


def convert_images_to_png(texture_dir: Path):
    texture_dir.mkdir(parents=True, exist_ok=True)

    converted_count = 0

    for index, image in enumerate(bpy.data.images):
        # Render ResultやViewer Nodeなどは対象外
        if image.type != "IMAGE":
            continue

        if image.size[0] <= 0 or image.size[1] <= 0:
            print(f"[WARN] Skip invalid image: {image.name}")
            continue

        safe_name = "".join(
            c if c.isalnum() or c in "-_" else "_"
            for c in image.name
        )

        if not safe_name:
            safe_name = f"texture_{index:03d}"

        png_path = texture_dir / f"{safe_name}_{index:03d}.png"

        print(f"[INFO] Converting image: {image.name}")
        print(f"[INFO] Output PNG     : {png_path}")

        # Blender内の展開済み画像をPNGとして保存
        image.filepath_raw = str(png_path)
        image.file_format = "PNG"

        try:
            image.save()
        except Exception:
            # save()が使えないケースではsave_renderを試す
            image.save_render(str(png_path))

        # 同じImage datablockをPNGファイル参照に変更
        image.filepath = str(png_path)
        image.filepath_raw = str(png_path)
        image.source = "FILE"

        try:
            image.reload()
        except Exception as exc:
            print(f"[WARN] Reload failed for {image.name}: {exc}")

        converted_count += 1

    print(f"[INFO] Converted images: {converted_count}")

    if converted_count == 0:
        raise RuntimeError("No valid images were converted.")


def main():
    args = parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()

    if args.texture_dir:
        texture_dir = Path(args.texture_dir).resolve()
    else:
        texture_dir = output_path.parent / f"{output_path.stem}_textures_png"

    if not input_path.exists():
        raise FileNotFoundError(f"Input GLB not found: {input_path}")

    output_path.parent.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"[INFO] Input GLB   : {input_path}")
    print(f"[INFO] Output GLB  : {output_path}")
    print(f"[INFO] Texture dir : {texture_dir}")
    print("=" * 60)

    clear_scene()

    bpy.ops.import_scene.gltf(filepath=str(input_path))

    convert_images_to_png(texture_dir)

    # Unity/glTFastで扱いやすい通常のGLBとして再出力
    bpy.ops.export_scene.gltf(
        filepath=str(output_path),
        export_format="GLB",
        export_image_format="AUTO",
        export_materials="EXPORT",
    )

    print(f"[SAVED] Unity-compatible GLB: {output_path}")


if __name__ == "__main__":
    main()