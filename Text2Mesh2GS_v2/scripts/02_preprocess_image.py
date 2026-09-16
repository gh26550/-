from pathlib import Path
import yaml
from PIL import Image
from rembg import remove


def load_config(config_path: str = "configs/pipeline.yaml"):
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_output_dirs(config):
    raw_image_dir = Path(config.get("output", {}).get("raw_image_dir", "outputs/images_raw"))
    preprocessed_image_dir = Path(
        config.get("output", {}).get("preprocessed_image_dir", "outputs/images_preprocessed")
    )
    return raw_image_dir, preprocessed_image_dir


def crop_to_alpha_bbox(img: Image.Image) -> Image.Image:
    bbox = img.getbbox()
    if bbox is None:
        raise ValueError("No visible object found after background removal.")
    return img.crop(bbox)


def place_on_square_canvas(img: Image.Image, padding_ratio: float) -> Image.Image:
    width, height = img.size
    max_side = max(width, height)

    canvas_size = int(max_side * (1.0 + padding_ratio * 2.0))
    canvas = Image.new("RGBA", (canvas_size, canvas_size), (0, 0, 0, 0))

    x = (canvas_size - width) // 2
    y = (canvas_size - height) // 2

    canvas.paste(img, (x, y), img)
    return canvas


def preprocess_one_image(input_path: Path, output_path: Path, output_resolution: int, padding_ratio: float):
    print(f"[INFO] Preprocessing: {input_path}")

    img = Image.open(input_path).convert("RGBA")

    # 背景除去
    img = remove(img)

    # オブジェクト領域でクロップ
    img = crop_to_alpha_bbox(img)

    # 正方形キャンバスに中央配置
    img = place_on_square_canvas(img, padding_ratio)

    # TRELLIS.2入力用に解像度を統一
    img = img.resize((output_resolution, output_resolution), Image.LANCZOS)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(output_path)

    print(f"[SAVED] {output_path}")


def main():
    config = load_config()

    preprocess_cfg = config.get("preprocess", {})
    output_resolution = int(preprocess_cfg.get("output_resolution", 512))
    padding_ratio = float(preprocess_cfg.get("padding_ratio", 0.15))

    raw_image_dir, preprocessed_image_dir = get_output_dirs(config)

    selected_images = sorted(raw_image_dir.glob("*/selected.png"))

    if len(selected_images) == 0:
        raise FileNotFoundError(
            "No selected.png files found. "
            "Please select one generated image per object and save it as selected.png."
        )

    for selected_path in selected_images:
        object_dir_name = selected_path.parent.name
        output_path = preprocessed_image_dir / object_dir_name / "input_rgba.png"

        preprocess_one_image(
            input_path=selected_path,
            output_path=output_path,
            output_resolution=output_resolution,
            padding_ratio=padding_ratio
        )

    print("[DONE] Image preprocessing completed.")


if __name__ == "__main__":
    main()