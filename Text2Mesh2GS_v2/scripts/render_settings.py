"""Shared checkpoint rendering metadata; importable without torch/gsplat."""
import math

KEYS = ("scale_min", "scale_max", "background_color", "image_downscale", "use_zbuffer_visibility", "zbuffer_depth_threshold")


def settings_from_args(args):
    result = {key: getattr(args, key) for key in KEYS}
    validate_settings(result)
    return result


def validate_settings(settings):
    lo, hi = settings["scale_min"], settings["scale_max"]
    if not math.isfinite(lo) or not math.isfinite(hi) or not 0 < lo <= hi:
        raise ValueError("Expected finite 0 < scale_min <= scale_max")
    if settings.get("image_downscale", 1) < 1:
        raise ValueError("image_downscale must be positive")


def inherit_settings(args, checkpoint, argv):
    saved = checkpoint.get("render_settings", {})
    flags = {arg.split("=", 1)[0] for arg in argv if arg.startswith("--")}
    for key in KEYS:
        if hasattr(args, key) and key in saved and "--" + key not in flags:
            setattr(args, key, saved[key])
    if getattr(args, "no_zbuffer_visibility", False):
        args.use_zbuffer_visibility = False
    validate_settings({"scale_min": args.scale_min, "scale_max": args.scale_max,
                       "image_downscale": getattr(args, "image_downscale", 1)})
    return saved
