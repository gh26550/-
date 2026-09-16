from pathlib import Path
import argparse
import yaml
import pandas as pd
import torch
from diffusers import StableDiffusionXLPipeline


DEFAULT_CONFIG_PATH = "configs/pipeline.yaml"


def load_config(config_path: str = DEFAULT_CONFIG_PATH):
    """Load pipeline.yaml."""
    config_path = Path(config_path)
    if not config_path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(config_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    if config is None:
        raise ValueError(f"Config file is empty: {config_path}")

    return config


def get_image_generation_config(config: dict) -> dict:
    """Read Stable Diffusion settings from image_generation section."""
    if "image_generation" not in config:
        raise KeyError("Missing 'image_generation' section in pipeline.yaml")

    image_cfg = config["image_generation"]

    if not image_cfg.get("enabled", True):
        raise RuntimeError("image_generation.enabled is false. Please enable it in pipeline.yaml.")

    required_keys = [
        "model_id",
        "prompt_template",
        "negative_prompt",
        "seed",
    ]

    for key in required_keys:
        if key not in image_cfg:
            raise KeyError(f"Missing image_generation.{key} in pipeline.yaml")

    # width / height がない場合は resolution を使う
    if "width" not in image_cfg:
        if "resolution" not in image_cfg:
            raise KeyError("Missing image_generation.width or image_generation.resolution in pipeline.yaml")
        image_cfg["width"] = image_cfg["resolution"]

    if "height" not in image_cfg:
        if "resolution" not in image_cfg:
            raise KeyError("Missing image_generation.height or image_generation.resolution in pipeline.yaml")
        image_cfg["height"] = image_cfg["resolution"]

    # 省略時のデフォルト値
    image_cfg.setdefault("num_inference_steps", 30)
    image_cfg.setdefault("guidance_scale", 7.5)
    image_cfg.setdefault("num_images_per_prompt", 4)

    return image_cfg


def get_output_root(config: dict) -> Path:
    """Read output.raw_image_dir. If missing, use project.output_root/images_raw."""
    if "output" in config and "raw_image_dir" in config["output"]:
        return Path(config["output"]["raw_image_dir"])

    output_root = config.get("project", {}).get("output_root", "outputs")
    return Path(output_root) / "images_raw"


def load_sdxl_pipeline(model_id: str):
    print(f"[INFO] Loading Stable Diffusion model: {model_id}")

    pipe = StableDiffusionXLPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16,
        use_safetensors=True,
    )

    pipe = pipe.to("cuda")

    # GPUメモリが厳しい場合に有効
    pipe.enable_attention_slicing()

    return pipe


def build_prompt(row: pd.Series, image_cfg: dict) -> tuple[str, str, str]:
    """Create object_id, object_name, and prompt from prompts.csv and prompt_template."""
    if "object_name" in row and pd.notna(row["object_name"]):
        object_name = str(row["object_name"])
    else:
        raise KeyError("prompts.csv must contain an 'object_name' column")

    if "id" in row and pd.notna(row["id"]):
        object_id = str(row["id"]).zfill(3)
    else:
        object_id = "000"

    prompt_template = str(image_cfg["prompt_template"])
    prompt = prompt_template.format(object_name=object_name)

    return object_id, object_name, prompt


def generate_images(pipe, row: pd.Series, config: dict):
    image_cfg = get_image_generation_config(config)

    object_id, object_name, prompt = build_prompt(row, image_cfg)
    negative_prompt = str(image_cfg["negative_prompt"])

    width = int(image_cfg["width"])
    height = int(image_cfg["height"])
    num_inference_steps = int(image_cfg["num_inference_steps"])
    guidance_scale = float(image_cfg["guidance_scale"])
    num_images_per_prompt = int(image_cfg["num_images_per_prompt"])
    base_seed = int(image_cfg["seed"])

    output_root = get_output_root(config)
    safe_object_name = object_name.replace(" ", "_").replace("/", "_")
    out_dir = output_root / f"{object_id}_{safe_object_name}"
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print(f"[INFO] Object ID   : {object_id}")
    print(f"[INFO] Object Name : {object_name}")
    print(f"[INFO] Prompt      : {prompt}")
    print(f"[INFO] Output Dir  : {out_dir}")
    print(f"[INFO] Seed        : {base_seed}")

    generator = torch.Generator(device="cuda").manual_seed(base_seed)

    result = pipe(
        prompt=prompt,
        negative_prompt=negative_prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        guidance_scale=guidance_scale,
        num_images_per_prompt=num_images_per_prompt,
        generator=generator,
    )

    for i, image in enumerate(result.images):
        save_path = out_dir / f"generated_{i:02d}.png"
        image.save(save_path)
        print(f"[SAVED] {save_path}")

    prompt_log = out_dir / "prompt.txt"
    with open(prompt_log, "w", encoding="utf-8") as f:
        f.write("[PROMPT]\n")
        f.write(prompt + "\n\n")
        f.write("[NEGATIVE PROMPT]\n")
        f.write(negative_prompt + "\n\n")
        f.write("[SETTINGS]\n")
        f.write(f"model_id: {image_cfg['model_id']}\n")
        f.write(f"width: {width}\n")
        f.write(f"height: {height}\n")
        f.write(f"num_inference_steps: {num_inference_steps}\n")
        f.write(f"guidance_scale: {guidance_scale}\n")
        f.write(f"num_images_per_prompt: {num_images_per_prompt}\n")
        f.write(f"seed: {base_seed}\n")

    print(f"[SAVED] {prompt_log}")


def parse_args():
    parser = argparse.ArgumentParser(description="Generate front-view object images using Stable Diffusion XL.")
    parser.add_argument(
        "--config",
        type=str,
        default=DEFAULT_CONFIG_PATH,
        help="Path to pipeline.yaml",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    config = load_config(args.config)
    image_cfg = get_image_generation_config(config)

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Please check your GPU/PyTorch environment.")

    if "input" not in config or "prompt_csv" not in config["input"]:
        raise KeyError("Missing input.prompt_csv in pipeline.yaml")

    prompt_csv = Path(config["input"]["prompt_csv"])
    if not prompt_csv.exists():
        raise FileNotFoundError(f"Prompt CSV not found: {prompt_csv}")

    df = pd.read_csv(prompt_csv)

    model_id = str(image_cfg["model_id"])
    pipe = load_sdxl_pipeline(model_id)

    for _, row in df.iterrows():
        generate_images(pipe, row, config)

    print("[DONE] Front-view image generation completed.")


if __name__ == "__main__":
    main()
