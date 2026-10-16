#!/usr/bin/env python3
"""Run the packaged correctness fixture; requires repo sources and Hexagon libnative."""
import argparse
import gzip
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import tempfile


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--hexagon-tools", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    archive = Path(__file__).resolve().parent
    if archive == args.output.resolve() or archive in args.output.resolve().parents:
        parser.error("output must be outside the immutable evidence archive")
    dependencies = json.loads((archive / "source/host-dependencies.json").read_text())
    for relative, expected in dependencies.items():
        path = args.repo / relative
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != expected:
            parser.error("frozen host dependency changed: " + relative)
    with tempfile.TemporaryDirectory(prefix="nntr-kv-fixed-tail-source-") as temporary:
        sources = {}
        for kind in ("baseline", "corrected"):
            source = Path(temporary) / (kind + "-hexkl_kv_q.c")
            source.write_bytes(gzip.decompress((archive / "source" / kind / "hexkl_kv_q.c.gz").read_bytes()))
            sources[kind] = source
        command = [sys.executable, str(archive / "tests/run_kv_fixed_tail_check.py"),
                   "--repo", str(args.repo.resolve()),
                   "--source", str(sources["corrected"]),
                   "--reference", str(sources["baseline"]),
                   "--hexagon-tools", str(args.hexagon_tools.resolve()),
                   "--output", str(args.output.resolve())]
        subprocess.run(command, check=True)


if __name__ == "__main__":
    main()
