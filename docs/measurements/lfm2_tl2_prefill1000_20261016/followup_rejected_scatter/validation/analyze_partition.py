#!/usr/bin/env python3
"""Exhaustive row ownership proof plus normal-route reconstruction counts."""
import argparse
import json
from pathlib import Path


def old_ranges(m, n):
    return [(m * i // n, m * (i + 1) // n) for i in range(n)]


def lane_ranges(m, n, i):
    if m < 4 * n:
        return [old_ranges(m, n)[i]]
    bulk = (m // n) & ~3
    start, tail = bulk * n, m - bulk * n
    return [(bulk * i, bulk * (i + 1)),
            (start + tail * i // n, start + tail * (i + 1) // n)]


p = argparse.ArgumentParser(description=__doc__)
p.add_argument("--histogram", type=Path,
               default=Path(__file__).resolve().with_name("route_block_histogram.json"))
p.add_argument("--output", type=Path, required=True)
a = p.parse_args()
hist = json.loads(a.histogram.read_text())
checks = 0
for m in range(65):
    for n in range(1, 65):
        got = []
        for i, (lo, hi) in enumerate(old_ranges(m, n)):
            current = [r for x, y in lane_ranges(m, n, i) for r in range(x, y)]
            assert len(current) == hi - lo
            got.extend(current)
        assert sorted(got) == list(range(m))
        checks += 1
result = {"partition_property_cases": checks, "range": "m0..64, lanes1..64",
          "exact_lane_row_counts": True, "disjoint_complete_row_coverage": True,
          "load_groups": {}}
for n in range(3, 9):
    sums, singles = [0, 0], [0, 0]
    for count, freq in hist.items():
        m = int(count)
        nt = min(n, m)
        old = old_ranges(m, nt)
        new = [r for i in range(nt) for r in lane_ranges(m, nt, i)]
        for k, ranges in enumerate([old, new]):
            for lo, hi in ranges:
                groups = max(0, hi // 4 - (lo + 3) // 4)
                fragment = hi - lo - 4 * groups
                sums[k] += (groups + fragment) * freq
                singles[k] += fragment * freq
    result["load_groups"][str(n)] = {
        "old": sums[0], "new": sums[1],
        "reduction_pct": 100 * (sums[0] - sums[1]) / sums[0],
        "old_single": singles[0], "new_single": singles[1]}
a.output.write_text(json.dumps(result, indent=2) + "\n")
print(f"ROW_PARTITION_PASS cases={checks}")
