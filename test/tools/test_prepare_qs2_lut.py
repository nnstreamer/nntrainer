# SPDX-License-Identifier: Apache-2.0
"""Run directly with Python; requires NumPy and native C/C++ compilers."""
import importlib.util
from pathlib import Path
import struct
import subprocess
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
TOOL = ROOT / "tools/htp/prepare_qs2_lut.py"
spec = importlib.util.spec_from_file_location("prepare_qs2", TOOL)
prepare = importlib.util.module_from_spec(spec)
spec.loader.exec_module(prepare)


class PreparedQs2Test(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory()
        cls.work = Path(cls.temp.name)
        backend = ROOT / "nntrainer/tensor/htp_backend"
        obj = cls.work / "kernel.o"
        subprocess.run(["cc", "-std=c11", "-O2", "-c", str(backend / "hvx/hvx_gemv_u8i2_lut.c"), "-o", str(obj)], check=True)
        cls.reader = cls.work / "reader"
        subprocess.run(["c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-I"+str(backend), "-I"+str(backend / "hvx"), str(ROOT / "test/htp/host/prepared_qs2_check.cpp"), str(obj), "-o", str(cls.reader)], check=True)

        cls.validator = cls.work / "validator"
        subprocess.run(["c++", "-std=c++17", "-O2", "-Wall", "-Wextra", "-Werror", "-I"+str(backend), str(ROOT / "tools/htp/validate_qs2_lut.cpp"), "-o", str(cls.validator)], check=True)

    @classmethod
    def tearDownClass(cls):
        cls.temp.cleanup()

    def fixture(self, k=64, n=256):
        codes = np.random.default_rng(42).integers(0, 256, k*n//4, dtype=np.uint8).tobytes()
        source = codes + struct.pack("4b", -8, -3, 0, 7) + np.ones(2*n, dtype="<f4").tobytes()
        model = self.work / "model.bin"
        artifact = self.work / "0.nntri2"
        model.write_bytes(source)
        artifact.write_bytes(prepare.encode(source, 0, k, n))
        return model, artifact, k, n

    def read(self, fixture):
        return subprocess.run([str(self.reader), *map(str, fixture)], capture_output=True)

    def test_real_shapes_cross_language(self):
        for k, n in [(32, 128), (64, 256), (2048, 3072), (1536, 2048)]:
            with self.subTest(K=k, N=n):
                result = self.read(self.fixture(k, n))
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_header_payload_and_truncation_rejected(self):
        f = self.fixture()
        original = f[1].read_bytes()
        for offset in [0, 8, 12, 16, 24, 32, 40, 44, 48, 56, 64, len(original)-1]:
            with self.subTest(offset=offset):
                data = bytearray(original)
                data[offset] ^= 1
                f[1].write_bytes(data)
                self.assertNotEqual(self.read(f).returncode, 0)
        f[1].write_bytes(original[:-1])
        self.assertNotEqual(self.read(f).returncode, 0)

    def test_stale_source_and_missing_file_rejected(self):
        f = self.fixture()
        data = bytearray(f[0].read_bytes())
        data[-1] ^= 1
        f[0].write_bytes(data)
        self.assertNotEqual(self.read(f).returncode, 0)
        f[1].unlink()
        self.assertNotEqual(self.read(f).returncode, 0)

    def test_cache_invalidates_modified_files(self):
        for target in ["source", "artifact"]:
            with self.subTest(target=target):
                f = self.fixture()
                result = subprocess.run([str(self.reader), *map(str, f), target], capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)

    def test_receipt_and_invalidation(self):
        import os
        for target in ["model", "artifact", "receipt"]:
            with self.subTest(target=target):
                f = self.fixture()
                subprocess.run([str(self.validator), str(f[0]), str(self.work)], check=True, capture_output=True)
                result = subprocess.run([str(self.reader), *map(str, f), "receipt"], capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
                if target == "receipt":
                    (self.work / ".nntri2.validated").write_bytes(b"truncated")
                else:
                    path = f[0] if target == "model" else f[1]
                    st = path.stat()
                    os.utime(path, ns=(st.st_atime_ns, st.st_mtime_ns + 1000000))
                result = subprocess.run([str(self.reader), *map(str, f), "no-receipt"], capture_output=True)
                self.assertEqual(result.returncode, 0, result.stderr)
        f = self.fixture()
        data = bytearray(f[1].read_bytes()); data[80] ^= 1; f[1].write_bytes(data)
        self.assertNotEqual(subprocess.run([str(self.validator), str(f[0]), str(self.work)], capture_output=True).returncode, 0)

    def test_cli_and_verify(self):
        import sys
        f = self.fixture()
        output = self.work / "output"
        cmd = [sys.executable, str(TOOL), "--model", str(f[0]), "--weight", "0:64:256", "--output", str(output)]
        subprocess.run(cmd, check=True, capture_output=True)
        subprocess.run(cmd + ["--verify-only"], check=True, capture_output=True)
        self.assertEqual((output / "0_64_256.nntri2").read_bytes(), f[1].read_bytes())
        log = self.work / "export.log"
        log.write_text("[QS2-WEIGHT] 0 64 256\n[QS2-WEIGHT] 0 64 256\n")
        subprocess.run([sys.executable, str(TOOL), "--model", str(f[0]),
                        "--manifest-log", str(log), "--output", str(output),
                        "--verify-only"], check=True, capture_output=True)
        self.assertNotEqual(subprocess.run(cmd + ["--weight", "0:64:256"], capture_output=True).returncode, 0)
        self.assertNotEqual(subprocess.run(cmd + ["--weight", "999999:32:128"], capture_output=True).returncode, 0)


if __name__ == "__main__":
    unittest.main()
