#!/usr/bin/env python3
"""V2 entry point: common settings, explicit execution, shared output paths."""
import argparse
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pipeline import read, write, gaussian_plan, execute


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--jobs", required=True)
    parser.add_argument("--project_root", default=".")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--config", default=str(Path(__file__).resolve().parents[1] / "configs/gaussian.json"))
    parser.add_argument("--execute", action="store_true")
    args = parser.parse_args()
    plan = gaussian_plan(read(args.jobs), read(args.config), args.project_root)
    write(Path(args.output_dir) / "plan.json", plan)
    if args.execute:
        execute(plan, args.project_root, args.output_dir)
    else:
        print("Plan saved; --execute required for rendering/training.")


if __name__ == "__main__":
    main()
