import argparse
import random
from render_settings import settings_from_args
import json
import math
from pathlib import Path

import imageio.v3 as iio
import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm
import trimesh

from gsplat import rasterization


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument("--multiview_dir", required=True)
    parser.add_argument("--mesh",  type=str,default="",help="Fallback GLB mesh path. Prefer --mesh_npz.",)
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--mesh_npz",type=str,default="",help="Blender Z-up mesh NPZ exported by 05.",)

    parser.add_argument("--num_points", type=int, default=80000)
    parser.add_argument("--steps", type=int, default=9000)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--image_downscale", type=int, default=2)
    parser.add_argument("--save_every", type=int, default=500)

    parser.add_argument("--lr_means", type=float, default=5e-7)
    parser.add_argument("--lr_scales", type=float, default=3e-4)
    parser.add_argument("--lr_quats", type=float, default=1e-4)
    parser.add_argument("--lr_opacities", type=float, default=4e-2)
    parser.add_argument("--lr_colors", type=float, default=8e-2)

    parser.add_argument("--anchor_weight", type=float, default=2.0)
    parser.add_argument("--opacity_reg_weight", type=float, default=2e-4)
    parser.add_argument("--scale_reg_weight", type=float, default=5e-4)

    parser.add_argument("--use_alpha_loss", action="store_true")
    parser.add_argument("--alpha_weight", type=float, default=30.0)

    parser.add_argument("--use_silhouette_loss", action="store_true")
    parser.add_argument("--silhouette_weight", type=float, default=0.2)

    parser.add_argument("--use_projection_loss", action="store_true")
    parser.add_argument("--projection_weight", type=float, default=5.0)
    parser.add_argument("--projection_mask_dilate", type=int, default=5)

    parser.add_argument("--init_scale", type=float, default=0.0015)
    parser.add_argument("--scale_min", type=float, default=0.00005)
    parser.add_argument("--scale_max", type=float, default=0.005)
    parser.add_argument("--init_opacity_logit", type=float, default=-2.2)

    # 追加：Mesh法線ベースの異方性Gaussian初期化
    parser.add_argument(
        "--use_anisotropic_init",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Initialize Gaussian orientation and scale from mesh surface normals.",
    )
    parser.add_argument(
        "--init_tangent_scale",
        type=float,
        default=0.0015,
        help="Initial scale along mesh tangent directions.",
    )
    parser.add_argument(
        "--init_normal_scale",
        type=float,
        default=0.00015,
        help="Initial scale along mesh normal direction. Should be much smaller than tangent scale.",
    )
    parser.add_argument(
        "--normal_offset",
        type=float,
        default=0.0,
        help="Optional offset along mesh normals. Usually 0.0 is safest.",
    )

    parser.add_argument("--align_mesh_to_render_space", action="store_true")

    parser.add_argument(
        "--background_color",
        type=str,
        default="0.770588,0.771569,0.770588",
        help="RGB background color used for compositing transparent renders.",
    )

    parser.add_argument("--init_color_from_views", action="store_true")
    parser.add_argument("--color_alpha_threshold", type=float, default=0.3)
    parser.add_argument(
        "--color_default",
        type=str,
        default="0.75,0.45,0.32",
        help="Default RGB color for points not observed in alpha foreground.",
    )

    # 追加：外部depthを使わず、Gaussian中心からviewごとのz-bufferを作る
    parser.add_argument("--use_zbuffer_visibility", action="store_true")
    parser.add_argument("--zbuffer_depth_threshold", type=float, default=0.01)

    return parser.parse_args()


def parse_background_color(text: str):
    vals = [float(v.strip()) for v in text.split(",")]
    if len(vals) != 3:
        raise ValueError("--background_color must be like '0.77,0.77,0.77'")
    return np.array(vals, dtype=np.float32)


def parse_rgb(text: str):
    vals = [float(v.strip()) for v in text.split(",")]
    if len(vals) != 3:
        raise ValueError("RGB must be like '0.75,0.45,0.32'")
    return np.array(vals, dtype=np.float32)


def load_images_and_cameras(
    multiview_dir: Path,
    downscale: int,
    device: str,
    background_color: np.ndarray,
):
    image_dir = multiview_dir / "images"
    camera_json = multiview_dir / "cameras" / "camera_params.json"

    if not image_dir.exists():
        raise FileNotFoundError(f"Image directory not found: {image_dir}")

    if not camera_json.exists():
        raise FileNotFoundError(f"camera_params.json not found: {camera_json}")

    with open(camera_json, "r", encoding="utf-8") as f:
        camera_data = json.load(f)

    views = camera_data["views"]

    images = []
    alphas = []
    backgrounds = []
    viewmats = []
    Ks = []
    filenames = []

    bg_np = background_color.reshape(1, 1, 3)

    for view in views:
        filename = view["filename"]
        img_path = image_dir / filename

        if not img_path.exists():
            raise FileNotFoundError(f"Image not found: {img_path}")

        img = iio.imread(img_path)

        if img.ndim != 3:
            raise ValueError(f"Unsupported image shape: {img.shape}, file: {img_path}")

        if img.shape[-1] == 4:
            rgb = img[..., :3].astype(np.float32) / 255.0
            alpha = img[..., 3:4].astype(np.float32) / 255.0

            img_rgb = rgb * alpha + bg_np * (1.0 - alpha)
            alpha_img = alpha.astype(np.float32)

        else:
            img_rgb = img[..., :3].astype(np.float32) / 255.0
            alpha_img = np.ones((*img_rgb.shape[:2], 1), dtype=np.float32)

        if downscale > 1:
            h, w = img_rgb.shape[:2]
            new_h = h // downscale
            new_w = w // downscale

            t_rgb = torch.from_numpy(img_rgb).permute(2, 0, 1).unsqueeze(0)
            t_alpha = torch.from_numpy(alpha_img).permute(2, 0, 1).unsqueeze(0)

            t_rgb = F.interpolate(t_rgb, size=(new_h, new_w), mode="area")
            t_alpha = F.interpolate(t_alpha, size=(new_h, new_w), mode="area")

            img_rgb = t_rgb.squeeze(0).permute(1, 2, 0).numpy()
            alpha_img = t_alpha.squeeze(0).permute(1, 2, 0).numpy()

        h, w = img_rgb.shape[:2]

        K = build_intrinsic_matrix(view, w, h)
        viewmat = blender_camera_matrix_to_gsplat_viewmat(view["matrix_world"])

        images.append(torch.from_numpy(img_rgb).float())
        alphas.append(torch.from_numpy(alpha_img).float())
        backgrounds.append(torch.from_numpy(background_color).float())
        viewmats.append(torch.from_numpy(viewmat).float())
        Ks.append(torch.from_numpy(K).float())
        filenames.append(filename)

    images = torch.stack(images, dim=0).to(device)
    alphas = torch.stack(alphas, dim=0).to(device)
    backgrounds = torch.stack(backgrounds, dim=0).to(device)
    viewmats = torch.stack(viewmats, dim=0).to(device)
    Ks = torch.stack(Ks, dim=0).to(device)

    return images, alphas, backgrounds, viewmats, Ks, filenames


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


def blender_camera_matrix_to_gsplat_viewmat(matrix_world):
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


def normalize_np(v, eps=1e-8):
    norm = np.linalg.norm(v, axis=-1, keepdims=True)
    return v / np.maximum(norm, eps)


def build_tangent_frames_from_normals(normals: np.ndarray):
    """
    法線nをlocal Z軸として、local X/Yを接線方向にする回転行列を作る。
    Rの列が [tangent1, tangent2, normal] になる。
    """
    n = normalize_np(normals.astype(np.float32))

    ref_z = np.array([0.0, 0.0, 1.0], dtype=np.float32)
    ref_y = np.array([0.0, 1.0, 0.0], dtype=np.float32)

    use_z = np.abs(n[:, 2]) < 0.9
    refs = np.where(use_z[:, None], ref_z[None, :], ref_y[None, :])

    t1 = np.cross(refs, n)
    t1 = normalize_np(t1)

    t2 = np.cross(n, t1)
    t2 = normalize_np(t2)

    R = np.stack([t1, t2, n], axis=-1).astype(np.float32)
    return R


def rotation_matrix_to_quaternion_wxyz(R: np.ndarray):
    """
    R: [N, 3, 3], local-to-world rotation matrix.
    return: [N, 4] quaternion in [w, x, y, z].
    """
    quats = np.zeros((R.shape[0], 4), dtype=np.float32)

    for i in range(R.shape[0]):
        m = R[i]
        trace = m[0, 0] + m[1, 1] + m[2, 2]

        if trace > 0.0:
            s = math.sqrt(trace + 1.0) * 2.0
            qw = 0.25 * s
            qx = (m[2, 1] - m[1, 2]) / s
            qy = (m[0, 2] - m[2, 0]) / s
            qz = (m[1, 0] - m[0, 1]) / s

        elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            s = math.sqrt(max(1.0 + m[0, 0] - m[1, 1] - m[2, 2], 1e-8)) * 2.0
            qw = (m[2, 1] - m[1, 2]) / s
            qx = 0.25 * s
            qy = (m[0, 1] + m[1, 0]) / s
            qz = (m[0, 2] + m[2, 0]) / s

        elif m[1, 1] > m[2, 2]:
            s = math.sqrt(max(1.0 + m[1, 1] - m[0, 0] - m[2, 2], 1e-8)) * 2.0
            qw = (m[0, 2] - m[2, 0]) / s
            qx = (m[0, 1] + m[1, 0]) / s
            qy = 0.25 * s
            qz = (m[1, 2] + m[2, 1]) / s

        else:
            s = math.sqrt(max(1.0 + m[2, 2] - m[0, 0] - m[1, 1], 1e-8)) * 2.0
            qw = (m[1, 0] - m[0, 1]) / s
            qx = (m[0, 2] + m[2, 0]) / s
            qy = (m[1, 2] + m[2, 1]) / s
            qz = 0.25 * s

        q = np.array([qw, qx, qy, qz], dtype=np.float32)
        q = q / max(np.linalg.norm(q), 1e-8)
        quats[i] = q

    return quats

def load_mesh_points_from_npz(
    mesh_npz_path: Path,
    num_points: int,
    normal_offset: float,
):
    """
    05で保存したBlender Z-up座標のvertices/facesから表面点を生成する。
    """
    if not mesh_npz_path.exists():
        raise FileNotFoundError(
            f"Blender mesh NPZ not found: {mesh_npz_path}"
        )

    data = np.load(mesh_npz_path, allow_pickle=False)

    if "vertices" not in data or "faces" not in data:
        raise KeyError(
            "Mesh NPZ must contain 'vertices' and 'faces'."
        )

    vertices = data["vertices"].astype(np.float32)
    faces = data["faces"].astype(np.int64)

    mesh = trimesh.Trimesh(
        vertices=vertices,
        faces=faces,
        process=False,
        validate=False,
    )

    if len(mesh.faces) == 0:
        raise RuntimeError("NPZ mesh has no faces.")

    points, face_indices = mesh.sample(
        num_points,
        return_index=True,
    )

    normals = mesh.face_normals[face_indices].astype(np.float32)
    normals = normalize_np(normals)

    if normal_offset != 0.0:
        points = points + normals * normal_offset

    # 色は後でview-based initializationにより上書きする
    colors = np.ones((num_points, 3), dtype=np.float32) * 0.7

    bbox_min = vertices.min(axis=0)
    bbox_max = vertices.max(axis=0)
    bbox_size = bbox_max - bbox_min

    print(f"[INFO] Loaded Blender mesh NPZ: {mesh_npz_path}")
    print(f"[INFO] NPZ vertices           : {len(vertices)}")
    print(f"[INFO] NPZ faces              : {len(faces)}")
    print(f"[INFO] NPZ bbox min           : {bbox_min}")
    print(f"[INFO] NPZ bbox max           : {bbox_max}")
    print(f"[INFO] NPZ bbox size          : {bbox_size}")

    return (
        points.astype(np.float32),
        colors.astype(np.float32),
        normals.astype(np.float32),
    )

def load_mesh_points(
    mesh_path: Path,
    num_points: int,
    align_to_render_space: bool,
    normal_offset: float,
):
    loaded = trimesh.load(mesh_path, force="scene")

    if isinstance(loaded, trimesh.Trimesh):
        mesh = loaded

    elif isinstance(loaded, trimesh.Scene):
        meshes = []

        for geom_name, geom in loaded.geometry.items():
            if not isinstance(geom, trimesh.Trimesh):
                continue

            transform = loaded.graph.get(geom_name)[0]
            geom_copy = geom.copy()
            geom_copy.apply_transform(transform)
            meshes.append(geom_copy)

        if len(meshes) == 0:
            raise RuntimeError(f"No mesh found in: {mesh_path}")

        mesh = trimesh.util.concatenate(meshes)

    else:
        raise RuntimeError(f"Unsupported mesh type: {type(loaded)}")

    if len(mesh.faces) == 0:
        raise RuntimeError("Mesh has no faces.")

    points, face_indices = mesh.sample(num_points, return_index=True)

    normals = mesh.face_normals[face_indices].astype(np.float32)
    normals = normalize_np(normals)

    if normal_offset != 0.0:
        points = points + normals * normal_offset

    colors = np.ones((num_points, 3), dtype=np.float32) * 0.7

    try:
        if hasattr(mesh.visual, "vertex_colors") and mesh.visual.vertex_colors is not None:
            vertex_colors = np.asarray(mesh.visual.vertex_colors[:, :3], dtype=np.float32) / 255.0

            if len(vertex_colors) == len(mesh.vertices):
                faces = mesh.faces[face_indices]
                colors = vertex_colors[faces].mean(axis=1)
    except Exception:
        pass

    if align_to_render_space:
        min_corner = mesh.bounds[0]
        max_corner = mesh.bounds[1]
        center = (min_corner + max_corner) * 0.5
        min_z = min_corner[2]

        translation = np.array(
            [-center[0], -center[1], -min_z],
            dtype=np.float32,
        )
        points = points + translation

    return (
        points.astype(np.float32),
        colors.astype(np.float32),
        normals.astype(np.float32),
    )


def init_gaussians(
    mesh_path: Path | None,
    mesh_npz_path: Path | None,
    num_points: int,
    device: str,
    init_scale: float,
    init_opacity_logit: float,
    align_to_render_space: bool,
    use_anisotropic_init: bool,
    init_tangent_scale: float,
    init_normal_scale: float,
    normal_offset: float,
):
    if mesh_npz_path is not None:
        points, colors, normals = load_mesh_points_from_npz(
            mesh_npz_path=mesh_npz_path,
            num_points=num_points,
            normal_offset=normal_offset,
        )
    else:
        if mesh_path is None:
            raise ValueError(
                "Either --mesh_npz or --mesh must be specified."
        )

        points, colors, normals = load_mesh_points(
            mesh_path=mesh_path,
            num_points=num_points,
            align_to_render_space=align_to_render_space,
            normal_offset=normal_offset,
        )

    points_tensor = torch.from_numpy(points).to(device)

    means = torch.nn.Parameter(points_tensor.clone())
    means_init = points_tensor.clone().detach()

    if use_anisotropic_init:
        R = build_tangent_frames_from_normals(normals)
        quats_np = rotation_matrix_to_quaternion_wxyz(R)

        scales_np = np.zeros((num_points, 3), dtype=np.float32)
        scales_np[:, 0] = init_tangent_scale
        scales_np[:, 1] = init_tangent_scale
        scales_np[:, 2] = init_normal_scale

        print("[INFO] anisotropic init: enabled")
        print(f"[INFO] init_tangent_scale: {init_tangent_scale}")
        print(f"[INFO] init_normal_scale : {init_normal_scale}")

    else:
        quats_np = np.tile(
            np.array([[1.0, 0.0, 0.0, 0.0]], dtype=np.float32),
            (num_points, 1),
        )

        scales_np = np.ones((num_points, 3), dtype=np.float32) * init_scale

        print("[INFO] anisotropic init: disabled")
        print(f"[INFO] isotropic init_scale: {init_scale}")

    log_scales = torch.nn.Parameter(
        torch.log(torch.from_numpy(scales_np).to(device))
    )

    quats = torch.nn.Parameter(
        torch.from_numpy(quats_np).to(device)
    )

    opacity_logits = torch.nn.Parameter(
        torch.ones((num_points,), device=device) * init_opacity_logit
    )

    colors = torch.clamp(
        torch.from_numpy(colors).to(device),
        1e-4,
        1.0 - 1e-4,
    )
    color_logits = torch.nn.Parameter(torch.logit(colors))

    return {
        "means": means,
        "means_init": means_init,
        "log_scales": log_scales,
        "quats": quats,
        "opacity_logits": opacity_logits,
        "color_logits": color_logits,
    }


def initialize_colors_from_views(
    points,
    images,
    alphas,
    viewmats,
    Ks,
    default_color,
    alpha_threshold,
    device,
):
    num_points = points.shape[0]
    num_views, height, width, _ = images.shape

    color_sum = torch.zeros((num_points, 3), device=device)
    color_count = torch.zeros((num_points, 1), device=device)

    points_h = torch.cat(
        [points, torch.ones((num_points, 1), device=device)],
        dim=1,
    )

    for vidx in range(num_views):
        viewmat = viewmats[vidx]
        K = Ks[vidx]

        cam = (viewmat @ points_h.T).T[:, :3]

        z = cam[:, 2]
        valid = z > 1e-6

        x = cam[:, 0]
        y = cam[:, 1]

        u = K[0, 0] * (x / z) + K[0, 2]
        v = K[1, 1] * (y / z) + K[1, 2]

        valid = valid & (u >= 0) & (u < width) & (v >= 0) & (v < height)

        if valid.sum() == 0:
            continue

        valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)

        u_i = u[valid].long()
        v_i = v[valid].long()

        a = alphas[vidx, v_i, u_i, 0]
        fg = a > alpha_threshold

        if fg.sum() == 0:
            continue

        point_ids = valid_idx[fg]
        rgb = images[vidx, v_i[fg], u_i[fg], :]

        color_sum[point_ids] += rgb
        color_count[point_ids] += 1.0

    default = torch.tensor(default_color, dtype=torch.float32, device=device).view(1, 3)

    colors = color_sum / torch.clamp(color_count, min=1.0)
    colors = torch.where(color_count > 0, colors, default.expand_as(colors))
    colors = torch.clamp(colors, 1e-4, 1.0 - 1e-4)

    valid_ratio = float((color_count > 0).float().mean().detach().cpu())
    print(f"[INFO] view-based color valid ratio: {valid_ratio:.6f}")
    print(f"[INFO] view-based color mean       : {colors.mean(dim=0).detach().cpu().numpy()}")

    return colors



def compute_zbuffer_visibility_mask(
    points,
    viewmat,
    K,
    width,
    height,
    depth_threshold,
):
    """
    外部depth mapを使わず、Gaussian中心から簡易z-bufferを作る。
    同じ画素に投影された点のうち、最も手前に近い点だけを可視にする。

    points: [N, 3]
    return: [N] float visibility mask
    """
    with torch.no_grad():
        device = points.device
        num_points = points.shape[0]

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

        # PyTorch 2.x
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


def make_projection_masks(points, viewmats, Ks, height, width, dilate_kernel, device):
    masks = []

    points_h = torch.cat(
        [points, torch.ones((points.shape[0], 1), device=device)],
        dim=1,
    )

    for viewmat, K in zip(viewmats, Ks):
        cam = (viewmat @ points_h.T).T[:, :3]

        z = cam[:, 2]
        valid = z > 1e-6

        x = cam[:, 0]
        y = cam[:, 1]

        u = K[0, 0] * (x / z) + K[0, 2]
        v = K[1, 1] * (y / z) + K[1, 2]

        valid = valid & (u >= 0) & (u < width) & (v >= 0) & (v < height)

        u = u[valid].long()
        v = v[valid].long()

        mask = torch.zeros((height, width, 1), device=device)

        if u.numel() > 0:
            mask[v, u, 0] = 1.0

        if dilate_kernel > 1:
            k = dilate_kernel
            pad = k // 2
            m = mask.permute(2, 0, 1).unsqueeze(0)
            m = F.max_pool2d(m, kernel_size=k, stride=1, padding=pad)
            mask = m.squeeze(0).permute(1, 2, 0)

        masks.append(mask)

    masks = torch.stack(masks, dim=0)
    return masks


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


def render_one_view(
    params,
    viewmat,
    K,
    bg_color,
    width,
    height,
    scale_min,
    scale_max,
    visibility_mask=None,
):
    means = params["means"]

    scales = torch.exp(params["log_scales"])
    scales = torch.clamp(scales, scale_min, scale_max)

    quats = F.normalize(params["quats"], dim=-1)
    opacities = torch.sigmoid(params["opacity_logits"])

    if visibility_mask is not None:
        opacities = opacities * visibility_mask.to(opacities.device)

    colors = torch.sigmoid(params["color_logits"])

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

    if alpha is not None:
        if alpha.dim() == 4:
            alpha_img = alpha[0]
        else:
            alpha_img = alpha

        if alpha_img.shape[-1] != 1:
            alpha_img = alpha_img[..., :1]
    else:
        alpha_img = None

    if (not used_backgrounds) and (alpha_img is not None):
        render = render * alpha_img + bg_color.view(1, 1, 3) * (1.0 - alpha_img)

    render = torch.clamp(render, 0.0, 1.0)

    return render, alpha_img


def save_checkpoint(params, output_dir: Path, step: int, render_settings):
    output_dir.mkdir(parents=True, exist_ok=True)

    ckpt = {
        "step": step,
        "schema_version": "2.0",
        "render_settings": render_settings,
        "means": params["means"].detach().cpu(),
        "means_init": params["means_init"].detach().cpu(),
        "log_scales": params["log_scales"].detach().cpu(),
        "quats": params["quats"].detach().cpu(),
        "opacity_logits": params["opacity_logits"].detach().cpu(),
        "color_logits": params["color_logits"].detach().cpu(),
    }

    path = output_dir / f"ckpt_{step:06d}.pt"
    torch.save(ckpt, path)
    print(f"[SAVED] {path}")


def save_training_config(args, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    config_path = output_dir / "train_config.json"

    with open(config_path, "w", encoding="utf-8") as f:
        json.dump(vars(args), f, indent=2)

    print(f"[SAVED] {config_path}")


def main():
    args = parse_args()
    render_settings = settings_from_args(args)
    if args.steps < 1 or args.num_points < 1 or args.save_every < 1:
        raise ValueError("steps, num_points and save_every must be positive")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    multiview_dir = Path(args.multiview_dir)
    mesh_path = Path(args.mesh) if args.mesh else None
    mesh_npz_path = Path(args.mesh_npz) if args.mesh_npz else None

    if mesh_npz_path is None and mesh_path is None:
        raise ValueError(
            "Specify --mesh_npz or --mesh."
        )
    output_dir = Path(args.output_dir)

    background_color = parse_background_color(args.background_color)

    print("=" * 60)
    print(f"[INFO] device                    : {device}")
    print(f"[INFO] multiview_dir             : {multiview_dir}")
    print(f"[INFO] mesh                      : {mesh_path}")
    print(f"[INFO] mesh_npz                  : {mesh_npz_path}")
    print(f"[INFO] output_dir                : {output_dir}")
    print(f"[INFO] num_points                : {args.num_points}")
    print(f"[INFO] steps                     : {args.steps}")
    print(f"[INFO] image_downscale           : {args.image_downscale}")
    print(f"[INFO] use_alpha_loss            : {args.use_alpha_loss}")
    print(f"[INFO] alpha_weight              : {args.alpha_weight}")
    print(f"[INFO] use_silhouette_loss       : {args.use_silhouette_loss}")
    print(f"[INFO] silhouette_weight         : {args.silhouette_weight}")
    print(f"[INFO] use_projection_loss       : {args.use_projection_loss}")
    print(f"[INFO] background_color          : {background_color}")
    print(f"[INFO] align_mesh_to_render_space: {args.align_mesh_to_render_space}")
    print(f"[INFO] use_anisotropic_init      : {args.use_anisotropic_init}")
    print(f"[INFO] init_tangent_scale        : {args.init_tangent_scale}")
    print(f"[INFO] init_normal_scale         : {args.init_normal_scale}")
    print(f"[INFO] normal_offset             : {args.normal_offset}")
    print(f"[INFO] init_color_from_views     : {args.init_color_from_views}")
    print(f"[INFO] color_alpha_threshold     : {args.color_alpha_threshold}")
    print(f"[INFO] color_default             : {args.color_default}")
    print(f"[INFO] use_zbuffer_visibility    : {args.use_zbuffer_visibility}")
    print(f"[INFO] zbuffer_depth_threshold   : {args.zbuffer_depth_threshold}")
    print("=" * 60)

    save_training_config(args, output_dir)

    images, alphas, backgrounds, viewmats, Ks, filenames = load_images_and_cameras(
        multiview_dir=multiview_dir,
        downscale=args.image_downscale,
        device=device,
        background_color=background_color,
    )

    num_views, height, width, _ = images.shape

    print(f"[INFO] views      : {num_views}")
    print(f"[INFO] image size : {width} x {height}")
    print(f"[INFO] bg mean    : {backgrounds.mean(dim=0).detach().cpu().numpy()}")
    print(f"[INFO] alpha mean : {alphas.mean().item():.6f}")
    print(f"[INFO] alpha min  : {alphas.min().item():.6f}")
    print(f"[INFO] alpha max  : {alphas.max().item():.6f}")

    params = init_gaussians(
        mesh_path=mesh_path,
        mesh_npz_path=mesh_npz_path,
        num_points=args.num_points,
        device=device,
        init_scale=args.init_scale,
        init_opacity_logit=args.init_opacity_logit,
        align_to_render_space=args.align_mesh_to_render_space,
        use_anisotropic_init=args.use_anisotropic_init,
        init_tangent_scale=args.init_tangent_scale,
        init_normal_scale=args.init_normal_scale,
        normal_offset=args.normal_offset,
    )

    if args.init_color_from_views:
        default_color = parse_rgb(args.color_default)

        init_colors = initialize_colors_from_views(
            points=params["means_init"],
            images=images,
            alphas=alphas,
            viewmats=viewmats,
            Ks=Ks,
            default_color=default_color,
            alpha_threshold=args.color_alpha_threshold,
            device=device,
        )

        with torch.no_grad():
            params["color_logits"].copy_(torch.logit(init_colors))

    save_checkpoint(params, output_dir, 0, render_settings)

    projection_masks = None
    if args.use_projection_loss:
        projection_masks = make_projection_masks(
            points=params["means_init"],
            viewmats=viewmats,
            Ks=Ks,
            height=height,
            width=width,
            dilate_kernel=args.projection_mask_dilate,
            device=device,
        )
        print(f"[INFO] projection mask mean: {projection_masks.mean().item():.6f}")

    optim = torch.optim.Adam(
        [
            {"params": [params["means"]], "lr": args.lr_means},
            {"params": [params["log_scales"]], "lr": args.lr_scales},
            {"params": [params["quats"]], "lr": args.lr_quats},
            {"params": [params["opacity_logits"]], "lr": args.lr_opacities},
            {"params": [params["color_logits"]], "lr": args.lr_colors},
        ]
    )

    pbar = tqdm(range(1, args.steps + 1))

    for step in pbar:
        idx = torch.randint(0, num_views, (1,), device=device).item()

        target = images[idx]
        target_alpha = alphas[idx]
        bg_color = backgrounds[idx]
        viewmat = viewmats[idx]
        K = Ks[idx]

        visibility_mask = None
        if args.use_zbuffer_visibility:
            visibility_mask = compute_zbuffer_visibility_mask(
                points=params["means_init"],
                viewmat=viewmat,
                K=K,
                width=width,
                height=height,
                depth_threshold=args.zbuffer_depth_threshold,
            )

        pred, pred_alpha = render_one_view(
            params=params,
            viewmat=viewmat,
            K=K,
            bg_color=bg_color,
            width=width,
            height=height,
            scale_min=args.scale_min,
            scale_max=args.scale_max,
            visibility_mask=visibility_mask,
        )

        eps = 1e-6

        if args.use_alpha_loss:
            fg = target_alpha
            loss_rgb = (((pred - target) ** 2) * fg).sum() / (fg.sum() * 3.0 + eps)

        elif args.use_projection_loss and projection_masks is not None:
            mask = projection_masks[idx]
            loss_rgb = (((pred - target) ** 2) * mask).sum() / (mask.sum() * 3.0 + eps)

        else:
            loss_rgb = F.mse_loss(pred, target)

        if args.use_silhouette_loss and pred_alpha is not None:
            fg = target_alpha
            bg = 1.0 - target_alpha

            loss_sil_fg = (((pred_alpha - 1.0) ** 2) * fg).sum() / (fg.sum() + eps)
            loss_sil_bg = ((pred_alpha ** 2) * bg).sum() / (bg.sum() + eps)

            loss_sil = loss_sil_fg + 0.05 * loss_sil_bg
        else:
            loss_sil_fg = torch.zeros((), device=device)
            loss_sil_bg = torch.zeros((), device=device)
            loss_sil = torch.zeros((), device=device)

        # Explicit alpha and outside-projection penalties. Their CLI weights now affect the loss.
        loss_alpha = F.mse_loss(pred_alpha, target_alpha) if args.use_alpha_loss and pred_alpha is not None else torch.zeros((), device=device)
        loss_projection = torch.zeros((), device=device)
        if args.use_projection_loss and projection_masks is not None and pred_alpha is not None:
            outside = 1.0 - projection_masks[idx]
            loss_projection = (pred_alpha.square() * outside).sum() / (outside.sum() + eps)

        loss_anchor = ((params["means"] - params["means_init"]) ** 2).mean()

        opacities = torch.sigmoid(params["opacity_logits"])
        scales = torch.exp(params["log_scales"])

        loss_opacity = opacities.mean()
        loss_scale = scales.mean()

        loss = (
            loss_rgb
            + args.alpha_weight * loss_alpha
            + args.projection_weight * loss_projection
            + args.silhouette_weight * loss_sil
            + args.anchor_weight * loss_anchor
            + args.opacity_reg_weight * loss_opacity
            + args.scale_reg_weight * loss_scale
        )

        optim.zero_grad()
        loss.backward()
        optim.step()

        with torch.no_grad():
            opacity_mean = opacities.mean().item()
            scale_mean = scales.mean().item()
            alpha_mean = pred_alpha.mean().item() if pred_alpha is not None else -1.0
            vis_mean = visibility_mask.mean().item() if visibility_mask is not None else -1.0

        pbar.set_description(
            f"loss={loss.item():.6f} "
            f"rgb={loss_rgb.item():.6f} "
            f"sil={loss_sil.item():.6f} "
            f"sfg={loss_sil_fg.item():.6f} "
            f"sbg={loss_sil_bg.item():.6f} "
            f"anchor={loss_anchor.item():.6f} "
            f"opa={opacity_mean:.4f} "
            f"scale={scale_mean:.5f} "
            f"a={alpha_mean:.4f} "
            f"vis={vis_mean:.4f}"
        )

        if step % args.save_every == 0:
            save_checkpoint(params, output_dir, step, render_settings)

    save_checkpoint(params, output_dir, args.steps, render_settings)

    print("[DONE] Training completed.")


if __name__ == "__main__":
    main()