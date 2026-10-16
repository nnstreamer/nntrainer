#!/usr/bin/env python3
"""Host parser/accounting fixtures and real supervisor lifecycle tests."""
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time
import unittest

MONITOR = Path(os.environ.get("MEMORY_MONITOR", Path(__file__).with_name("memory_monitor.host"))).resolve()


class MonitorTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory(prefix="nntr-monitor-test-")
        self.root = Path(self.tmp.name)
        self.proc = self.root / "proc"
        self.proc.mkdir()
        (self.proc / "fdinfo").mkdir()
        (self.proc / "status").write_text("Name:\tfixture\nRssAnon:\t1024 kB\nVmRSS:\t2048 kB\nVmHWM:\t4096 kB\n")
        self.out = self.root / "report.json"

    def tearDown(self):
        self.tmp.cleanup()

    def fd(self, number, text):
        (self.proc / "fdinfo" / str(number)).write_text(text)

    def snapshot(self):
        subprocess.run([str(MONITOR), "--output", str(self.out), "--snapshot-proc-dir", str(self.proc)], check=True)
        return json.loads(self.out.read_text())

    def run_child(self, code, expected=0):
        result = subprocess.run([str(MONITOR), "--output", str(self.out), "--period-ms", "20", "--", sys.executable, "-c", code], timeout=10)
        self.assertEqual(result.returncode, expected)
        return json.loads(self.out.read_text())

    def test_identity_dedup_raw_fallback_and_invalid_field(self):
        self.fd(0, "pos:\t0\nflags:\t01\nino:\t2\n")
        self.fd(1, "mnt_id:\t2\nino:\t7\nsize:\t4096\nexp_name:\tsystem\n")
        self.fd(2, "mnt_id:\t2\nino:\t7\nsize:\t4096\n")
        self.fd(3, "mnt_id:\t2\nino:\t8\nsize:\t8192\n")
        self.fd(4, "size:\t2048\n")
        self.fd(5, "mnt_id:\t3\nino:\t7\nsize:\t1024\n")
        self.fd(6, "size:\tnope\n")
        self.fd(7, "mnt_id:\t2\nino:\t9\nsize:\t0\n")
        report = self.snapshot()
        sample = report["peak_dedup_sample"]
        self.assertEqual(sample["fdinfo_size_bytes_raw"], 19456)
        self.assertEqual(sample["fdinfo_size_bytes_dedup_or_raw_fallback"], 15360)
        self.assertEqual(report["peak_anon_plus_fdinfo_dedup_or_raw_fallback_bytes"], 1063936)
        self.assertEqual(report["peak_anon_plus_fdinfo_raw_bytes"], 1068032)
        self.assertEqual(sample["duplicate_fds"], 1)
        self.assertEqual(sample["unique_known_buffers"], 4)
        self.assertEqual(sample["unknown_identity_fds"], 1)
        self.assertEqual(sample["fdinfo_exporter_evidence_fds"], 1)
        self.assertEqual(sample["invalid_fields"], 1)
        self.assertFalse(sample["dedup_complete_for_readable_size_fds"])
        self.assertFalse(report["dma_buffer_identity_verified"])

    def test_conflicting_inode_sizes_are_flagged(self):
        self.fd(1, "mnt_id: 2\nino: 7\nsize: 4096\n")
        self.fd(2, "mnt_id: 2\nino: 7\nsize: 8192\n")
        sample = self.snapshot()["peak_dedup_sample"]
        self.assertEqual(sample["fdinfo_size_bytes_raw"], 12288)
        self.assertEqual(sample["fdinfo_size_bytes_dedup_or_raw_fallback"], 8192)
        self.assertEqual(sample["inode_size_conflicts"], 1)
        self.assertFalse(sample["dedup_complete_for_readable_size_fds"])

    def test_missing_mount_identity_is_explicit(self):
        self.fd(1, "ino: 7\nsize: 4096\n")
        self.fd(2, "ino: 7\nsize: 4096\n")
        sample = self.snapshot()["peak_dedup_sample"]
        self.assertEqual(sample["fdinfo_size_bytes_dedup_or_raw_fallback"], 4096)
        self.assertEqual(sample["ino_without_mount_fds"], 2)
        self.assertFalse(sample["dedup_complete_for_readable_size_fds"])

    def test_missing_anon_yields_null_composite(self):
        (self.proc / "status").write_text("Name: zombie\n")
        report = self.snapshot()
        self.assertIsNone(report["peak_anon_plus_fdinfo_dedup_or_raw_fallback_bytes"])
        self.assertEqual(report["composite_available_sample_count"], 0)

    def test_unreadable_fd_is_reported(self):
        (self.proc / "fdinfo" / "5").symlink_to(self.root / "nonexistent")
        sample = self.snapshot()["peak_dedup_sample"]
        self.assertEqual(sample["fdinfo_read_errors"], 1)
        self.assertFalse(sample["dedup_complete_for_readable_size_fds"])

    def test_counter_overflow_does_not_emit_wrapped_total(self):
        self.fd(1, "size: 18446744073709551615\n")
        self.fd(2, "size: 1\n")
        report = self.snapshot()
        self.assertTrue(report["samples"][0]["counter_overflow"])
        self.assertIsNone(report["peak_anon_plus_fdinfo_raw_bytes"])

    def test_child_exit_and_allocation_lifetime(self):
        report = self.run_child("import time,sys; x=bytearray(32*1024*1024); time.sleep(0.2); sys.exit(7)", expected=7)
        self.assertEqual(report["child_exit_code"], 7)
        self.assertIsNone(report["child_signal"])
        self.assertTrue(report["samples"][0]["pre_exec"])
        self.assertGreaterEqual(report["sample_count"], 5)
        self.assertGreater(report["max_sampled_rss_anon_kib"], 30000)
        self.assertGreater(report["child_final_ru_maxrss_kib"], 30000)
        self.assertGreaterEqual(report["sample_interval_ms"]["min"], 19)
        peak = report["peak_dedup_sample"]
        self.assertEqual(report["peak_anon_plus_fdinfo_dedup_or_raw_fallback_bytes"], peak["rss_anon_kib"] * 1024 + peak["fdinfo_size_bytes_dedup_or_raw_fallback"])

    def test_exec_failure_exit_and_report(self):
        result = subprocess.run([str(MONITOR), "--output", str(self.out), "--", str(self.root / "not-a-command")], timeout=5, stderr=subprocess.PIPE)
        self.assertEqual(result.returncode, 127)
        self.assertEqual(json.loads(self.out.read_text())["child_exit_code"], 127)

    def test_signal_forward_and_child_reap(self):
        ready = self.root / "ready"
        script = "import os,time; from pathlib import Path; Path(" + repr(str(ready)) + ").write_text(str(os.getpid())); time.sleep(30)"
        process = subprocess.Popen([str(MONITOR), "--output", str(self.out), "--", sys.executable, "-c", script])
        try:
            deadline = time.monotonic() + 5
            while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.01)
            self.assertTrue(ready.exists())
            child_pid = int(ready.read_text())
            process.send_signal(signal.SIGTERM)
            self.assertEqual(process.wait(timeout=5), 143)
            report = json.loads(self.out.read_text())
            self.assertEqual(report["child_signal"], 15)
            self.assertEqual(report["forwarded_signal"], 15)
            self.assertFalse(Path("/proc", str(child_pid)).exists())
        finally:
            if process.poll() is None:
                process.kill()
                process.wait()

    def test_failed_output_does_not_execute_child(self):
        marker = self.root / "must-not-exist"
        result = subprocess.run([str(MONITOR), "--output", str(self.root / "missing" / "file"), "--", sys.executable, "-c", "from pathlib import Path; Path(" + repr(str(marker)) + ").touch()"], stderr=subprocess.PIPE)
        self.assertEqual(result.returncode, 1)
        self.assertFalse(marker.exists())


if __name__ == "__main__":
    unittest.main(verbosity=2)
