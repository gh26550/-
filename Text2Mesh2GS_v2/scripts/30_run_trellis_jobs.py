#!/usr/bin/env python3
"""Run an explicitly configured external TRELLIS CLI; do not guess a version-specific API."""
import argparse
import json
from pathlib import Path
import subprocess


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--jobs", required=True)
    p.add_argument("--command-config", required=True)
    p.add_argument("--project-root", default=".")
    p.add_argument("--output", required=True)
    p.add_argument("--execute", action="store_true")
    args = p.parse_args()
    jobs = json.loads(Path(args.jobs).read_text(encoding="utf-8-sig"))["jobs"]
    config = json.loads(Path(args.command_config).read_text(encoding="utf-8-sig"))
    template = config.get("command")
    if not isinstance(template, list) or not template or not all(isinstance(x, str) for x in template):
        raise ValueError("command must be an argument array for your installed TRELLIS CLI")
    if not any("{input_image}" in x for x in template) or not any("{expected_glb}" in x for x in template):
        raise ValueError("command must contain {input_image} and {expected_glb}")
    root = Path(args.project_root).resolve(); results = []
    output = Path(args.output); output.parent.mkdir(parents=True, exist_ok=True)
    for job in jobs:
        if not job.get("input_image"):
            raise ValueError(f"Missing image for {job.get('object_id')}")
        fields = dict(job)
        fields["input_image"] = str((root / job["input_image"]).resolve())
        fields["expected_glb"] = str((root / job["expected_glb"]).resolve())
        command = [part.format_map(fields) for part in template]
        row = {"object_id": job["object_id"], "command": command, "status": "planned"}
        results.append(row)
        if args.execute:
            if not Path(fields["input_image"]).is_file():
                raise ValueError("Image does not exist: " + fields["input_image"])
            Path(fields["expected_glb"]).parent.mkdir(parents=True, exist_ok=True)
            log = output.parent / (job["object_id"] + ".log")
            with log.open("w", encoding="utf-8") as stream:
                proc = subprocess.run(command, cwd=root, stdout=stream, stderr=subprocess.STDOUT, check=False)
            row.update(return_code=proc.returncode, log=str(log), status="completed" if proc.returncode==0 and Path(fields["expected_glb"]).is_file() else "failed")
        output.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
        if row["status"] == "failed":
            raise RuntimeError(f"TRELLIS output failed: {job['object_id']}")
    if not jobs:
        output.write_text("[]\n", encoding="utf-8")


if __name__ == "__main__":
    main()
