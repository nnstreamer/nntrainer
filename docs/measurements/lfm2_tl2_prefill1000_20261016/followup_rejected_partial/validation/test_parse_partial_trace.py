#!/usr/bin/env python3
"""Synthetic contract/mutation tests; does not modify or access a device."""
import contextlib
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import parse_partial_trace as parser


def trace(knob, token):
    return (f"10-16 20:13:58.001 123 124 E FastRPC: [LFM-PARTIAL] tok={token} rc=0 "
            f"rounds={2 if knob else 0} early_rows={2 if knob else 0} "
            f"early_batches={2 if knob else 0} final_first={1 if knob else 0} "
            f"ready_wait_us={70 if knob else 0} final_wait_us={30 if knob else 100}\n")


def make_run(stem, knob):
    ids = "".join(f"{i + 100}\n" for i in range(65))
    Path(str(stem) + ".ids").write_text(ids)
    log = (
        "prefill: 1024 tokens, 1280 ms, 800 TPS\n"
        "generation: 64 tokens, 4000 ms, 16 TPS\n"
        "[HTP] graph per-kind us/token: RMSNORM=1 FC=1 CONV1D_GATE=1 QK_NORM=1 "
        "ROPE=1 ATTN_M1=1 ADD=1 ROUTER_TOPK=1 MOE=1100(net of miss wait 1000) "
        "DENSE_FFN=1 LM_HEAD=1 | ops/wall=0.99 compute_mhz=2112\n"
        "[HTP] token driver: L0 us/token rt=1200 dsp_wall=1200 wake=0 "
        "hop_us=0 arm_fwd=1200 arm_us=0 arm_n=63\n"
        "[HTP] token driver: pool misses=256 misses/token=4 miss_wait_us/token=100 "
        "rounds=192 arm_ms/round=1 pgpgin_mib=2 rss_mib=3 mem_available_mib=4\n"
        "[HTP] token driver: round us: notice=0 pre=1 read=2 answer=3 (arm total) "
        "| dsp overlap=4 wait=5 anslat=6 rebind=7\n"
        "[HTP] token driver: expert_pair_reads=128\n"
        "[HTP] token driver: moe calls/token=5 (one-at-a-time 6)\n"
        "[HTP] token driver: close tokens=64 hops/token=0 served=64 timeouts=0 "
        "stale=0 id_checked=0 id_mismatch=0 stop_err=0x0\n"
    )
    if knob:
        log += "[HTP] token driver: partial_ready_rounds=128\n"
    Path(str(stem) + ".log").write_text(log)
    Path(str(stem) + ".logcat.txt").write_text("".join(trace(knob, i) for i in range(64)))
    meta = {
        "environment": {"NNTR_HTP_PAGE_CACHE": "0", "NNTR_MOE_CACHE_GROW": "0",
                        "NNTR_MOE_PRELOAD_EXPERTS": "128", "NNTR_HTP_PROFILE": "1",
                        "NNTR_LFM2_DECODE_PARTIAL_READY": str(knob)},
        "returncode": 0, "valid": True, "profile": True, "instrumented": True,
        "performance_comparison_eligible": False, "decode_tokens": 64,
        "decode_ms": 4000, "decode_tps": 16, "prefill_tokens": 1024,
        "prefill_ms": 1280, "prefill_tps": 800, "moe_cache_experts": 16,
        "ids_sha256": hashlib.sha256(ids.encode()).hexdigest(),
        "ids_match_reference": True, "concurrent_processes": [],
        "model_config_sha256": "a" * 64,
    }
    Path(str(stem) + ".json").write_text(json.dumps(meta))


class StrictParserCheck(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.off = Path(self.temp.name) / "off"
        self.on = Path(self.temp.name) / "on"
        make_run(self.off, 0)
        make_run(self.on, 1)

    def read(self):
        return parser.read_run(self.on, 1, 64, 0)

    def change(self, suffix, transform):
        p = Path(str(self.on) + suffix)
        p.write_text(transform(p.read_text()))

    def reject(self, suffix, transform, message):
        self.change(suffix, transform)
        result = self.read()
        self.assertFalse(result["valid"])
        self.assertIn(message, "\n".join(result["errors"]))
        return result

    def test_valid_full_pair_and_mirrored_sink(self):
        off, on = parser.read_run(self.off, 0, 64, 0), self.read()
        self.assertTrue(off["valid"], off["errors"])
        self.assertTrue(on["valid"], on["errors"])
        self.assertEqual(on["trace"]["aggregate"]["totals"]["early_rows"], 128)
        self.assertFalse(parser.compare(off, on)["errors"])
        cat = Path(str(self.on) + ".logcat.txt").read_text()
        self.change(".log", lambda x: x + cat)
        mirrored = self.read()
        self.assertTrue(mirrored["valid"], mirrored["errors"])
        self.assertEqual(mirrored["trace"]["aggregate"]["totals"]["early_rows"], 128)

    def test_missing_trace_never_becomes_zero(self):
        result = self.reject(".logcat.txt", lambda x: "unrelated log\n", "zero markers")
        self.assertIsNone(result["trace"]["aggregate"])
        self.assertEqual(result["summaries"]["graph_us_per_token"]["MOE_net_of_miss_wait"], 1000)

    def test_malformed_token(self):
        self.reject(".logcat.txt", lambda x: x.replace("tok=3 ", "tok=oops "), "malformed diagnostic")

    def test_missing_token(self):
        self.reject(".logcat.txt", lambda x: x.replace(trace(1, 3), ""), "got 63")

    def test_duplicate_token(self):
        self.reject(".logcat.txt", lambda x: x + trace(1, 3), "duplicate token")

    def test_out_of_order(self):
        self.reject(".logcat.txt", lambda x: x.replace(trace(1, 2) + trace(1, 3), trace(1, 3) + trace(1, 2)), "consecutive")

    def test_out_of_range(self):
        self.reject(".logcat.txt", lambda x: x.replace("tok=63 ", "tok=64 "), "consecutive")

    def test_unknown_field(self):
        self.reject(".logcat.txt", lambda x: x.replace("final_wait_us=30\n", "final_wait_us=30 extra=9\n", 1), "malformed diagnostic")

    def test_nonzero_return(self):
        self.reject(".logcat.txt", lambda x: x.replace("rc=0 ", "rc=1 ", 1), "rc=1")

    def test_bad_count_invariant(self):
        self.reject(".logcat.txt", lambda x: x.replace("early_rows=2 ", "early_rows=0 ", 1), "invalid rows/batches")

    def test_round_limit(self):
        self.reject(".logcat.txt", lambda x: x.replace("rounds=2 ", "rounds=39 ", 1), "invalid rounds/final_first")

    def test_conflicting_mirrored_sink(self):
        cat = Path(str(self.on) + ".logcat.txt").read_text()
        self.reject(".log", lambda x: x + cat.replace("ready_wait_us=70", "ready_wait_us=71"), "sinks disagree")

    def test_missing_summary(self):
        self.reject(".log", lambda x: "\n".join(l for l in x.splitlines() if "pool misses=" not in l), "expected one summary, got 0")

    def test_duplicate_summary(self):
        self.reject(".log", lambda x: x + "generation: 64 tokens, 4000 ms, 16 TPS\n", "expected one summary, got 2")

    def test_missing_net(self):
        self.reject(".log", lambda x: x.replace("(net of miss wait 1000)", ""), "missing net MoE")

    def test_net_mismatch(self):
        self.reject(".log", lambda x: x.replace("miss wait 1000", "miss wait 900"), "does not reconcile")

    def test_arm_publication_overflow(self):
        self.reject(".log", lambda x: x.replace("partial_ready_rounds=128", "partial_ready_rounds=129"), "ARM publications exceed")

    def test_malformed_optional_count(self):
        result = self.reject(".log", lambda x: x.replace("partial_ready_rounds=128", "partial_ready_rounds=bad"), "malformed summary")
        self.assertIsNone(result["summaries"]["arm_partial_ready_rounds"])

    def test_bad_ids(self):
        self.reject(".ids", lambda x: x + "oops\n", "decimal lines")

    def test_id_hash_mismatch(self):
        self.reject(".ids", lambda x: x.replace("100\n", "200\n", 1), "SHA256")

    def test_metadata_does_not_override_log(self):
        self.reject(".json", lambda x: x.replace('"decode_tps": 16', '"decode_tps": 17'), "differs between log and metadata")

    def test_failed_reference(self):
        self.reject(".json", lambda x: x.replace('"ids_match_reference": true', '"ids_match_reference": false'), "ids_match_reference")

    def test_cross_run_env_drift(self):
        meta_path = Path(str(self.on) + ".json")
        meta = json.loads(meta_path.read_text())
        meta["environment"]["NNTR_MOE_PREFETCH_READERS"] = "3"
        meta_path.write_text(json.dumps(meta))
        off, on = parser.read_run(self.off, 0, 64, 0), self.read()
        self.assertTrue(on["valid"])
        self.assertIn("environment differs", " ".join(parser.compare(off, on)["errors"]))

    def test_cli_missing_trace_writes_error_report(self):
        self.change(".logcat.txt", lambda x: "")
        output = Path(self.temp.name) / "report.json"
        with contextlib.redirect_stdout(io.StringIO()):
            rc = parser.main(["--off-stem", str(self.off), "--on-stem", str(self.on), "--output", str(output)])
        self.assertEqual(rc, 2)
        result = json.loads(output.read_text())
        self.assertFalse(result["valid"])
        self.assertIsNone(result["on"]["trace"]["aggregate"])
        self.assertIn("moe_net_us_per_token", result["comparison"]["deltas"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
