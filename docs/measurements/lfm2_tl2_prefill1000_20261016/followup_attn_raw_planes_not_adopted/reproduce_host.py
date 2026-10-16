#!/usr/bin/env python3
"""Run the archived strict helper fixture using a compatible nntrainer checkout.

Requires Linux, the recorded Hexagon 19.0.04 libnative tools, a C compiler and
the repository headers/stubs. It does not build/deploy a DSP or contact a device.
The supplied repository and archive remain read-only; generated files go to
--output, and decompressed frozen sources use a temporary directory.
"""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--archive", type=Path, default=Path(__file__).resolve().parent)
    p.add_argument("--repo", type=Path, required=True)
    p.add_argument("--hexagon-tools", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    root = a.archive.resolve()
    output = a.output.resolve()
    if output == root or root in output.parents:
        p.error("--output must be outside the archive")
    if sys.platform != "linux":
        p.error("The archived libnative stand-in fixture requires Linux")
    manifest = json.loads((root / "manifest.json").read_text())["files"]
    with tempfile.TemporaryDirectory(prefix="nntr-raw4-frozen-host-") as folder:
        temp = Path(folder)
        for rel, info in manifest.items():
            if not rel.startswith("source/"):
                continue
            data = (root / rel).read_bytes()
            if hashlib.sha256(data).hexdigest() != info["sha256"]:
                raise ValueError("Changed frozen source: " + rel)
            if rel.endswith(".gz"):
                data = gzip.decompress(data)
                rel = rel[:-3]
            target = temp / rel
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(data)
        candidate = temp / "source/candidate"
        cmd = [sys.executable, "-B", str(candidate / "tools/lfm2_attn_raw_planes_check.py"),
               "--repo", str(a.repo.resolve()), "--source",
               str(candidate / "nntrainer/tensor/htp_backend/hmx/hexkl_attn_q.c"),
               "--hexagon-tools", str(a.hexagon_tools.resolve()),
               "--output", str(output), "--sanitize"]
        output.mkdir(parents=True, exist_ok=True)
        (output / "archive-host-command.json").write_text(json.dumps(cmd, indent=2) + "\n")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
