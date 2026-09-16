#!/usr/bin/env python3
import argparse
import sys
import json
from render_settings import inherit_settings
from pathlib import Path
import numpy as np
import torch

SH_C0 = 0.28209479177387814
SH_REST_COUNT = 45


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument('--checkpoint', required=True)
    p.add_argument('--output', required=True)
    p.add_argument('--scale_min', type=float, default=0.00005)
    p.add_argument('--scale_max', type=float, default=0.005)
    p.add_argument('--opacity_threshold', type=float, default=0.0)
    p.add_argument('--max_gaussians', type=int, default=0)
    p.add_argument('--sort_by_opacity', action='store_true')
    return p.parse_args()


def sigmoid_np(x):
    x = np.clip(x, -80.0, 80.0)
    return 1.0 / (1.0 + np.exp(-x))


def normalize_quaternions_wxyz(q):
    return q / np.maximum(np.linalg.norm(q, axis=1, keepdims=True), 1e-12)


def write_binary_ply(path, means, log_scales, quats, opacity_logits, rgb):
    path.parent.mkdir(parents=True, exist_ok=True)
    n = means.shape[0]
    normals = np.zeros((n, 3), np.float32)
    f_dc = (rgb - 0.5) / SH_C0
    f_rest = np.zeros((n, SH_REST_COUNT), np.float32)

    fields = [
        ('x', means[:, 0]), ('y', means[:, 1]), ('z', means[:, 2]),
        ('nx', normals[:, 0]), ('ny', normals[:, 1]), ('nz', normals[:, 2]),
        ('f_dc_0', f_dc[:, 0]), ('f_dc_1', f_dc[:, 1]), ('f_dc_2', f_dc[:, 2]),
    ]
    for i in range(SH_REST_COUNT):
        fields.append((f'f_rest_{i}', f_rest[:, i]))
    fields += [
        ('opacity', opacity_logits),
        ('scale_0', log_scales[:, 0]), ('scale_1', log_scales[:, 1]), ('scale_2', log_scales[:, 2]),
        ('rot_0', quats[:, 0]), ('rot_1', quats[:, 1]), ('rot_2', quats[:, 2]), ('rot_3', quats[:, 3]),
    ]

    header = ['ply', 'format binary_little_endian 1.0', f'element vertex {n}']
    header += [f'property float {name}' for name, _ in fields]
    header += ['end_header']

    matrix = np.column_stack([np.asarray(v, np.float32) for _, v in fields]).astype('<f4', copy=False)
    with path.open('wb') as f:
        f.write(('\n'.join(header) + '\n').encode('ascii'))
        matrix.tofile(f)
    return matrix.shape[1]


def main():
    args = parse_args()
    ckpt_path = Path(args.checkpoint)
    out_path = Path(args.output)
    ckpt = torch.load(ckpt_path, map_location='cpu')
    inherit_settings(args, ckpt, sys.argv[1:])

    means = ckpt['means'].detach().float().cpu().numpy()
    log_scales = ckpt['log_scales'].detach().float().cpu().numpy()
    quats = normalize_quaternions_wxyz(ckpt['quats'].detach().float().cpu().numpy()).astype(np.float32)
    opacity_logits = ckpt['opacity_logits'].detach().float().cpu().numpy().reshape(-1)
    color_logits = ckpt['color_logits'].detach().float().cpu().numpy()
    rgb = sigmoid_np(color_logits).astype(np.float32)
    opacity = sigmoid_np(opacity_logits)

    keep = (
        np.isfinite(means).all(1) & np.isfinite(log_scales).all(1) &
        np.isfinite(quats).all(1) & np.isfinite(opacity_logits) &
        np.isfinite(rgb).all(1) & (opacity >= args.opacity_threshold)
    )
    # Preserve rejection of non-finite original values, then match the rasterizer's effective scale.
    log_scales = np.clip(log_scales, np.log(args.scale_min), np.log(args.scale_max))
    idx = np.flatnonzero(keep)
    if args.sort_by_opacity:
        idx = idx[np.argsort(opacity[idx])[::-1]]
    if args.max_gaussians > 0:
        idx = idx[:args.max_gaussians]
    if idx.size == 0:
        raise RuntimeError('No Gaussians remain after filtering.')

    props = write_binary_ply(
        out_path,
        means[idx].astype(np.float32),
        log_scales[idx].astype(np.float32),
        quats[idx].astype(np.float32),
        opacity_logits[idx].astype(np.float32),
        np.clip(rgb[idx], 0, 1).astype(np.float32),
    )

    metadata = {"schema_version": "2.0", "source_checkpoint": str(ckpt_path),
                "scale_min": args.scale_min, "scale_max": args.scale_max,
                "per_view_visibility_baked": False,
                "training_used_zbuffer_visibility": ckpt.get("render_settings", {}).get("use_zbuffer_visibility", False)}
    out_path.with_suffix(".metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    scales = np.exp(log_scales[idx])
    print('=' * 60)
    print(f'[SAVED] {out_path}')
    print(f'[INFO] Gaussian count      : {idx.size}')
    print(f'[INFO] PLY properties      : {props}')
    print(f'[INFO] opacity min/mean/max: {opacity[idx].min():.6f} / {opacity[idx].mean():.6f} / {opacity[idx].max():.6f}')
    print(f'[INFO] scale min/mean/max  : {scales.min():.6f} / {scales.mean():.6f} / {scales.max():.6f}')
    print(f'[INFO] bounds min          : {means[idx].min(0)}')
    print(f'[INFO] bounds max          : {means[idx].max(0)}')
    print('=' * 60)


if __name__ == '__main__':
    main()
