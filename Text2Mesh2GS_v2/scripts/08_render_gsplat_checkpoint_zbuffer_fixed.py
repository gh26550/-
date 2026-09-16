import argparse
import sys
from render_settings import inherit_settings
import json
import math
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F

from gsplat import rasterization


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--multiview_dir", required=True)
    parser.add_argument("--output_dir", required=True)

    parser.add_argument("--view_index", type=int, default=0)
    parser.add_argument("--image_downscale", type=int, default=2)

    parser.add_argument("--scale_min", type=float, default=0.0005)
    parser.add_argument("--scale_max", type=float, default=0.05)

    parser.add_argument(
        "--background_color",
        type=str,
        default="0.770588,0.771569,0.770588",
        help="RGB background color used for compositing transparent renders.",
    )

    # z-buffer visibility for rendering/checking.
    parser.add_argument("--use_zbuffer_visibility", action="store_true")
    parser.add_argument("--no_zbuffer_visibility", action="store_true", help="Validate portable PLY appearance without per-view masking")
    parser.add_argument("--zbuffer_depth_threshold", type=float, default=0.01)

    return parser.parse_args()


def parse_background_color(text: str):
    vals = [float(v.strip()) for v in text.split(",")]

    if len(vals) != 3:
        raise ValueError(
            "--background_color must be like '0.770588,0.771569,0.770588'"
        )

    return np.array(vals, dtype=np.float32)


def load_target_image(path: Path, downscale: int, background_color: np.ndarray):
    """
    Load target image.

    If the image is RGBA, compose it using the same background color as the
    training script. This keeps the target used by 07 and the target used by
    this 08 check script consistent.
    """
    img = iio.imread(path)

    if img.ndim != 3:
        raise ValueError(f"Unsupported image shape: {img.shape}, file: {path}")

    bg_np = background_color.reshape(1, 1, 3)

    if img.shape[-1] == 4:
        rgb = img[..., :3].astype(np.float32) / 255.0
        alpha = img[..., 3:4].astype(np.float32) / 255.0
        img_rgb = rgb * alpha + bg_np * (1.0 - alpha)
    else:
        img_rgb = img[..., :3].astype(np.float32) / 255.0
        alpha = np.ones((*img_rgb.shape[:2], 1), dtype=np.float32)

    if downscale > 1:
        h, w = img_rgb.shape[:2]
        new_h = h // downscale
        new_w = w // downscale

        t_rgb = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)
        t_alpha = torch.from_numpy(alpha).permute(2, 0, 1).unsqueeze(0)

        t_rgb = F.interpolate(t_rgb, size=(new_h, new_w), mode="area")
        t_alpha = F.interpolate(t_alpha, size=(new_h, new_w), mode="area")

        img_rgb = t_rgb.squeeze(0).permute(1, 2, 0).numpy()
        alpha = t_alpha.squeeze(0).permute(1, 2, 0).numpy()

    bg = background_color.astype(np.float32)

    return img_rgb.astype(np.float32), alpha.astype(np.float32), bg


def blender_camera_matrix_to_gsplat_viewmat(matrix_world):
    """
    Blender camera:
        - camera looks along -Z
        - Y is up

    gsplat / OpenCV-like camera:
        - camera looks along +Z
        - Y is down
    """
    c2w_blender = np.asarray(matrix_world, dtype=np.float64)
    w2c_blender = np.linalg.inv(c2w_blender)

    R_w2bcam = w2c_blender[:3, :3]
    t_w2bcam = w2c_blender[:3, 3]

    R_bcam_to_cv = np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, -1.0, 0.0],
            [0.0, 0.0, -1.0],
        ],
        dtype=np.float64,
    )

    R = R_bcam_to_cv @ R_w2bcam
    t = R_bcam_to_cv @ t_w2bcam

    viewmat = np.eye(4, dtype=np.float32)
    viewmat[:3, :3] = R.astype(np.float32)
    viewmat[:3, 3] = t.astype(np.float32)

    return viewmat


def build_intrinsic_matrix(view, width, height):
    if "angle_x" not in view or "angle_y" not in view:
        raise ValueError(
            "camera_params.json must contain angle_x and angle_y. "
            "Please render multiview images with perspective camera."
        )

    angle_x = float(view["angle_x"])
    angle_y = float(view["angle_y"])

    fx = 0.5 * width / math.tan(0.5 * angle_x)
    fy = 0.5 * height / math.tan(0.5 * angle_y)
    cx = width * 0.5
    cy = height * 0.5

    K = np.array(
        [
            [fx, 0.0, cx],
            [0.0, fy, cy],
            [0.0, 0.0, 1.0],
        ],
        dtype=np.float32,
    )

    return K


def compute_zbuffer_visibility_mask(
    points,
    viewmat,
    K,
    width,
    height,
    depth_threshold,
):
    """
    Build a simple z-buffer from Gaussian centers.

    For each pixel, the Gaussian center with the smallest camera-space z is
    treated as the front surface. Points within depth_threshold of that front
    z are visible. This is only a center-based approximation, but it is useful
    for checking whether the training-time z-buffer visibility changes the
    rendered appearance.

    Returns:
        visibility: [N] float tensor with 1.0 for visible Gaussians and 0.0 for hidden ones.
    """
    with torch.no_grad():
        device = points.device
        num_points = points.shape[0]

        viewmat = viewmat.to(device)
        K = K.to(device)

        points_h = torch.cat(
            [points, torch.ones((num_points, 1), device=device)],
            dim=1,
        )

        cam = (viewmat @ points_h.T).T[:, :3]

        z = cam[:, 2]
        valid = z > 1e-6

        x = cam[:, 0]
        y = cam[:, 1]

        u = K[0, 0] * (x / z) + K[0, 2]
        v = K[1, 1] * (y / z) + K[1, 2]

        valid = valid & (u >= 0) & (u < width) & (v >= 0) & (v < height)

        visibility = torch.zeros((num_points,), device=device)

        if valid.sum() == 0:
            return visibility

        valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)

        u_i = u[valid].long()
        v_i = v[valid].long()
        z_i = z[valid]

        pix = v_i * width + u_i

        zbuf = torch.full(
            (height * width,),
            float("inf"),
            dtype=z_i.dtype,
            device=device,
        )

        # PyTorch 2.x API.
        zbuf.scatter_reduce_(
            0,
            pix,
            z_i,
            reduce="amin",
            include_self=True,
        )

        z_front = zbuf[pix]
        visible_valid = z_i <= (z_front + depth_threshold)

        visibility[valid_idx[visible_valid]] = 1.0

        return visibility


def call_rasterization(
    means,
    quats,
    scales,
    opacities,
    colors,
    viewmats,
    Ks,
    width,
    height,
    backgrounds,
):
    """Handle gsplat API differences around the backgrounds argument."""
    try:
        outputs = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            backgrounds=backgrounds,
            packed=False,
        )
        used_backgrounds = True
        return outputs, used_backgrounds

    except TypeError as e:
        if "backgrounds" not in str(e):
            raise e

        print("[WARN] backgrounds argument is not supported by this gsplat version.")
        print("[WARN] Rendering without backgrounds. This may cause black background.")

        outputs = rasterization(
            means=means,
            quats=quats,
            scales=scales,
            opacities=opacities,
            colors=colors,
            viewmats=viewmats,
            Ks=Ks,
            width=width,
            height=height,
            packed=False,
        )
        used_backgrounds = False
        return outputs, used_backgrounds


def render_checkpoint(
    ckpt,
    viewmat,
    K,
    bg_color,
    width,
    height,
    scale_min,
    scale_max,
    device,
    use_zbuffer_visibility=False,
    zbuffer_depth_threshold=0.01,
):
    means = ckpt["means"].to(device)

    scales = torch.exp(ckpt["log_scales"].to(device))
    scales = torch.clamp(scales, scale_min, scale_max)

    quats = F.normalize(ckpt["quats"].to(device), dim=-1)
    opacities = torch.sigmoid(ckpt["opacity_logits"].to(device))
    colors = torch.sigmoid(ckpt["color_logits"].to(device))

    viewmat = viewmat.to(device)
    K = K.to(device)
    bg_color = bg_color.to(device)

    visibility_mean = -1.0
    visibility_count = -1
    visibility_mask = None

    if use_zbuffer_visibility:
        visibility_mask = compute_zbuffer_visibility_mask(
            points=means,
            viewmat=viewmat,
            K=K,
            width=width,
            height=height,
            depth_threshold=zbuffer_depth_threshold,
        )
        visibility_mean = float(visibility_mask.mean().detach().cpu())
        visibility_count = int(visibility_mask.sum().detach().cpu())
        print(f"[INFO] zbuffer visibility mean : {visibility_mean:.6f}")
        print(f"[INFO] zbuffer visible count   : {visibility_count} / {means.shape[0]}")
        opacities = opacities * visibility_mask

    viewmats = viewmat.unsqueeze(0)
    Ks = K.unsqueeze(0)
    backgrounds = bg_color.unsqueeze(0)

    outputs, used_backgrounds = call_rasterization(
        means=means,
        quats=quats,
        scales=scales,
        opacities=opacities,
        colors=colors,
        viewmats=viewmats,
        Ks=Ks,
        width=width,
        height=height,
        backgrounds=backgrounds,
    )

    render = None
    alpha = None

    if isinstance(outputs, tuple):
        if len(outputs) >= 1:
            render = outputs[0]
        if len(outputs) >= 2:
            alpha = outputs[1]
    else:
        render = outputs

    if render is None:
        raise RuntimeError("rasterization did not return render image.")

    if render.dim() == 4:
        render = render[0]

    render = render[..., :3]

    pred_alpha = None

    if alpha is not None:
        if alpha.dim() == 4:
            pred_alpha = alpha[0]
        else:
            pred_alpha = alpha

        if pred_alpha.shape[-1] != 1:
            pred_alpha = pred_alpha[..., :1]

    # If backgrounds argument was not supported, composite manually.
    if (not used_backgrounds) and (pred_alpha is not None):
        render = render * pred_alpha + bg_color.view(1, 1, 3) * (1.0 - pred_alpha)

    render = torch.clamp(render, 0.0, 1.0)

    means_cpu = means.detach().cpu()

    stats = {
        "visibility_mean": visibility_mean,
        "visibility_count": visibility_count,
        "visibility_total": int(means.shape[0]),
        "opacity_mean": float(opacities.mean().detach().cpu()),
        "opacity_min": float(opacities.min().detach().cpu()),
        "opacity_max": float(opacities.max().detach().cpu()),
        "scale_mean": float(scales.mean().detach().cpu()),
        "scale_min": float(scales.min().detach().cpu()),
        "scale_max": float(scales.max().detach().cpu()),
        "means_min": means_cpu.min(dim=0).values.numpy().tolist(),
        "means_max": means_cpu.max(dim=0).values.numpy().tolist(),
        "means_center": means_cpu.mean(dim=0).numpy().tolist(),
        "pred_alpha_mean": float(pred_alpha.mean().detach().cpu()) if pred_alpha is not None else -1.0,
        "pred_alpha_min": float(pred_alpha.min().detach().cpu()) if pred_alpha is not None else -1.0,
        "pred_alpha_max": float(pred_alpha.max().detach().cpu()) if pred_alpha is not None else -1.0,
    }

    return render.detach().cpu().numpy(), pred_alpha, stats


def save_image(path: Path, img: np.ndarray):
    path.parent.mkdir(parents=True, exist_ok=True)
    img8 = np.clip(img * 255.0, 0, 255).astype(np.uint8)
    iio.imwrite(path, img8)


def make_error_image(pred: np.ndarray, target: np.ndarray):
    error = np.abs(pred - target)
    error_vis = np.clip(error * 4.0, 0.0, 1.0)
    return error_vis


def main():
    args = parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    checkpoint_path = Path(args.checkpoint)
    multiview_dir = Path(args.multiview_dir)
    output_dir = Path(args.output_dir)

    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

    ckpt_settings = torch.load(checkpoint_path, map_location="cpu")
    inherit_settings(args, ckpt_settings, sys.argv[1:])

    camera_json = multiview_dir / "cameras" / "camera_params.json"
    image_dir = multiview_dir / "images"

    if not camera_json.exists():
        raise FileNotFoundError(f"camera_params.json not found: {camera_json}")

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    background_color = parse_background_color(args.background_color)

    with open(camera_json, "r", encoding="utf-8") as f:
        camera_data = json.load(f)

    views = camera_data["views"]

    if args.view_index < 0 or args.view_index >= len(views):
        raise IndexError(
            f"view_index out of range: {args.view_index}, "
            f"available: 0 to {len(views) - 1}"
        )

    view = views[args.view_index]
    target_path = image_dir / view["filename"]

    target, target_alpha, bg = load_target_image(
        path=target_path,
        downscale=args.image_downscale,
        background_color=background_color,
    )

    height, width = target.shape[:2]

    viewmat_np = blender_camera_matrix_to_gsplat_viewmat(view["matrix_world"])
    K_np = build_intrinsic_matrix(view, width, height)

    viewmat = torch.from_numpy(viewmat_np).float()
    K = torch.from_numpy(K_np).float()
    bg_color = torch.from_numpy(bg).float()

    ckpt = torch.load(checkpoint_path, map_location="cpu")

    pred, pred_alpha, stats = render_checkpoint(
        ckpt=ckpt,
        viewmat=viewmat,
        K=K,
        bg_color=bg_color,
        width=width,
        height=height,
        scale_min=args.scale_min,
        scale_max=args.scale_max,
        device=device,
        use_zbuffer_visibility=args.use_zbuffer_visibility,
        zbuffer_depth_threshold=args.zbuffer_depth_threshold,
    )

    error_vis = make_error_image(pred, target)
    comparison = np.concatenate([target, pred, error_vis], axis=1)

    output_dir.mkdir(parents=True, exist_ok=True)

    save_image(output_dir / f"target_{args.view_index:03d}.png", target)
    save_image(output_dir / f"render_{args.view_index:03d}.png", pred)
    save_image(output_dir / f"error_{args.view_index:03d}.png", error_vis)
    save_image(output_dir / f"compare_{args.view_index:03d}.png", comparison)

    if pred_alpha is not None:
        pred_alpha_np = pred_alpha.detach().cpu().numpy()
        save_image(
            output_dir / f"pred_alpha_{args.view_index:03d}.png",
            pred_alpha_np.repeat(3, axis=-1),
        )

    save_image(
        output_dir / f"target_alpha_{args.view_index:03d}.png",
        target_alpha.repeat(3, axis=-1),
    )

    mse = float(np.mean((pred - target) ** 2))

    print("=" * 60)
    print(f"[INFO] checkpoint          : {checkpoint_path}")
    print(f"[INFO] view index          : {args.view_index}")
    print(f"[INFO] target              : {target_path}")
    print(f"[INFO] image size          : {width} x {height}")
    print(f"[INFO] background color    : {background_color}")
    print(f"[INFO] use zbuffer         : {args.use_zbuffer_visibility}")
    print(f"[INFO] zbuffer threshold   : {args.zbuffer_depth_threshold}")
    print(f"[INFO] visibility mean     : {stats['visibility_mean']:.6f}")
    print(f"[INFO] visibility count    : {stats['visibility_count']} / {stats['visibility_total']}")
    print(f"[INFO] target alpha mean   : {target_alpha.mean():.6f}")
    print(f"[INFO] target alpha min/max: {target_alpha.min():.6f} / {target_alpha.max():.6f}")
    print(f"[INFO] pred alpha mean     : {stats['pred_alpha_mean']:.6f}")
    print(f"[INFO] pred alpha min/max  : {stats['pred_alpha_min']:.6f} / {stats['pred_alpha_max']:.6f}")
    print(f"[INFO] MSE                 : {mse:.8f}")
    print(f"[INFO] opacity mean        : {stats['opacity_mean']:.6f}")
    print(f"[INFO] opacity min/max     : {stats['opacity_min']:.6f} / {stats['opacity_max']:.6f}")
    print(f"[INFO] scale mean          : {stats['scale_mean']:.6f}")
    print(f"[INFO] scale min/max       : {stats['scale_min']:.6f} / {stats['scale_max']:.6f}")
    print(f"[INFO] means center        : {stats['means_center']}")
    print(f"[INFO] means min           : {stats['means_min']}")
    print(f"[INFO] means max           : {stats['means_max']}")
    print(f"[SAVED] compare            : {output_dir / f'compare_{args.view_index:03d}.png'}")
    print("=" * 60)


if __name__ == "__main__":
    main()
